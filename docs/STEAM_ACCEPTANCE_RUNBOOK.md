# Steam acceptance runbook

**Status: BLOCKED pending the user's authorized AppID and Steamworks SDK.** Nothing in this
repository has been validated against Valve's services. This document lists what the code does, what
the local tests do and do not prove, what must be provided by an authorized person, and how to record
evidence once those things exist.

Hard rules for whoever runs this:

- **App ID `3133700` is mock-only.** It resolves on the public Steam Store to someone else's product
  (a Fantasy Grounds DLC; see `docs/STEAM_STATUS.md`). Never initialize the SDK, upload a build, create
  stats, achievements, leaderboards or cloud files, or publish anything under it. It exists in this
  repo as a deterministic test constant (`src/hivemind/client/steamworks.py:24`).
- Accepting Valve agreements, paying the app fee, creating the partner account, generating Web API
  keys and signing in are actions only the authorized person performs, in their own browser. This
  runbook never asks an assistant or script to do them or to see the credentials.
- Do not commit the Steamworks SDK, `steam_api64.dll`, `steam_appid.txt` for a real app, Web API
  publisher keys, SteamID64s of other people, or auth-ticket bytes.

## 1. What the code does today

| Feature | What the repo actually does | Valve-verified? | Source |
|---|---|---|---|
| Library load | Searches the working directory, `steam_dist/HiveMind_Steam_FreeEdition[/steamworks_sdk]` and the repo root for `steam_api64.dll` / `libsteam_api.so` and loads it with `ctypes.CDLL`. | Only if a real library is present | `steamworks.py:139-180` |
| Init | Calls `SteamAPI_Init` if the library exports it and records the boolean in `native_steam_api_initialized`. | Needs Steam running + a valid AppID | `steamworks.py:172-176` |
| Rich presence | With native init true: calls `SteamAPI_SteamFriends_v017` then `SteamAPI_ISteamFriends_SetRichPresence(self, "status", text)`. Otherwise records an HMAC receipt. The accessor name is tied to one SDK interface version. | Native call only when initialized | `steamworks.py:238-252` |
| Auth ticket | `get_auth_session_ticket` returns an **HMAC-SHA256 over `steamid:appid:spend:vac`** keyed with a secret whose default (`b"HiveMind-Steamworks-Auth-Key"`) is committed to the repo. It is not an `ISteamUser::GetAuthSessionTicket` ticket and `account_spend_usd` is a caller-supplied number. | **No** | `steamworks.py:327-342`, `:344-366` |
| Leaderboard | Updates a Python dict and returns an HMAC receipt `steam_lb_hmac_...`. No `ISteamUserStats` leaderboard call. | **No** | `steamworks.py:411-448` |
| Achievements / overlay | Python list and queued dicts; no `SetAchievement` / `StoreStats`, no overlay hook. | **No** | `steamworks.py:280-326` |
| Cloud saves | `steam_edition.py` writes `steam_cloud/steam_remote_storage_profiles.json` on disk under an `ISteamRemoteStorage::...` namespace label. No `ISteamRemoteStorage` call. | **No** | `src/hivemind/game/steam_edition.py:1114-1134` |
| SteamPipe | Writes local VDF templates (`app_build_3133700.vdf`, ...). No upload happens. | **No** | `docs/STEAM_STATUS.md` |

## 2. What the local tests do and do not prove

| Test | Proves | Does not prove |
|---|---|---|
| `tests/test_daemon_steamworks.py::test_steamworks_sybil_resistance_and_leaderboard_credits` | Internal HMAC consistency: a ticket the same bridge signed verifies, and a ticket whose caller-supplied `account_spend_usd` is below the minimum is rejected as "Limited". Credit and achievement bookkeeping in Python state. | Anything about Valve tickets or real account standing. The signing secret is public, and the spend figure is an argument, so these tickets are forgeable by anyone who reads the repo. |
| `tests/test_steam_native_regressions.py` | With a **fake** native object, the rich-presence call passes the interface pointer as first argument, sets `argtypes`/`restype`, honors the boolean result, and does not call native code when init failed. Credit idempotency and input validation. | That a real `steam_api64.dll` exports those symbols, that the flat-API names match your SDK version, that Steam accepts the call, or that a friend sees the status. |
| `tests/test_daemon_steamworks.py::test_windows_idle_hook_and_game_launch_preemption` | `get_windows_idle_seconds()` returns a non-negative number on the local OS, and the daemon with simulated idle time and a simulated active game reports `should_yield_vram`. | Detection of a real game launched through Steam, or the real idle behaviour after input. |
| ctypes loading code path | Falls back cleanly when no library is present. | Behaviour of the real library, ABI mismatches, 32/64-bit mismatches. |

None of these start Steam, talk to Valve, or exercise cloud saves, the overlay, achievements or
leaderboards.

## 3. Prerequisites (all provided by the authorized person)

1. A Steamworks partner account and a **partner-assigned AppID for this project**, with the agreements
   accepted by the account owner. Record the AppID only in local, uncommitted configuration.
2. The Steamworks SDK downloaded by that person under Valve's license. Place the platform library
   next to the test executable or point `SteamworksSDKBridge(sdk_search_dir=...)` at it. Record the SDK
   version, because flat-API accessor names such as `SteamAPI_SteamFriends_v017` change between versions.
3. The Steam client running and signed in on the test machine as an account that owns or is
   authorized for the app (a developer/test account the owner controls).
4. For dev runs, a `steam_appid.txt` containing the real AppID beside the executable, kept out of git.
5. Steamworks Web API publisher key (for server-side ticket validation), stored by the owner in an
   environment variable or secret store. Never paste it into a command line, log, issue or this repo.
6. Partner-site configuration done by the owner: stats and achievements definitions, leaderboard
   definitions, Steam Cloud quotas and file paths, and any app launch options.

Without items 1-3 every step in section 4 stays `BLOCKED`.

## 4. Acceptance steps

For each step record: date, SDK version, Steam client build, OS, commit, command, observed result,
and a screenshot or log excerpt with identifiers redacted (section 5). Steps marked **needs code**
cannot pass until native calls exist; report them as `NOT IMPLEMENTED` and do not substitute the HMAC
receipt for a pass.

| # | Check | How | Pass evidence |
|---|---|---|---|
| 1 | Library found and initialized | Run a small script: `b = SteamworksSDKBridge(sdk_search_dir=Path(r"<sdk dir>")); print(b.get_native_bridge_status())` with Steam running. | `native_dll_loaded: true`, `steam_api_initialized: true`, `app_id` shows the configured AppID (note: `get_native_bridge_status` currently reports the constant 3133700, so also confirm the AppID from Steam itself). |
| 1b | Negative controls | Same script with Steam closed, then with a wrong/missing `steam_appid.txt`. | `steam_api_initialized: false` each time; no exception escapes. |
| 2 | Rich presence | With step 1 passing: `b.set_rich_presence("<own steamid64>", mission_name="Acceptance")`. Look at your own profile from a second account (friend). | `native_dll_broadcast` true **and** the friend sees the status text. A true flag alone is not enough. |
| 3 | Auth ticket (**needs code**) | Real flow: client `ISteamUser::GetAuthSessionTicket`, server validates with `ISteamUserAuth/AuthenticateUserTicket` or `ISteamGameServer::BeginAuthSession`. | Valve returns a verified SteamID for the ticket; a modified ticket is rejected by Valve. |
| 4 | Non-limited account / VAC checks (**needs code**) | Server-side lookup via Steam Web API for the validated SteamID. | Real account data, not the caller-supplied `account_spend_usd`. |
| 5 | Achievements (**needs code**) | `ISteamUserStats::SetAchievement` + `StoreStats` after stats are requested. | Achievement visible on the profile of the test account, survives restart. |
| 6 | Leaderboards (**needs code**) | `FindLeaderboard`, `UploadLeaderboardScore` (KeepBest), `DownloadLeaderboardEntries`, handling the async call results. | Entry appears for the test account, a lower score does not replace it. |
| 7 | Cloud saves (**needs code**) | `ISteamRemoteStorage::FileWrite` / `FileRead` on machine 1, sign in on machine 2 (or clear local data) and sync. | Same bytes on machine 2; conflict behaviour recorded; quota error path recorded. |
| 8 | Overlay (**needs code**) | Launch through Steam (or the owner's documented dev launch). Check `ISteamUtils::IsOverlayEnabled` and open the overlay with the shortcut; trigger a real achievement toast. | Overlay opens and the toast comes from Valve, not from the Python queue. |
| 9 | SteamPipe build/upload | Owner runs `steamcmd` with their own credentials against their own app and a non-default branch. | Build id on the partner site. This runbook never handles those credentials. |

Report a step as `PASS` only when the observable Valve-side effect is shown. A boolean from our own
Python layer, an HMAC receipt, or a JSON file is a local-state result.

## 5. Redaction

- SteamID64 (yours and any friend's): replace with `STEAMID-A`, `STEAMID-B`, or a truncated hash.
- AppID: keep out of public notes until the store page is announced. Say "the project AppID".
- Never include auth ticket hex, Web API keys, `steam_appid.txt` for a real app, cookies, `config.vdf`,
  or screenshots that show a username, email, friends list or wallet balance.
- SDK files are Valve-licensed: record the version, do not attach the binaries.

## 6. Results template

```
Steam acceptance result: <date>
Commit: <sha>   Operator: <role>   SDK version: <x>   Steam client build: <y>   OS: <z>
AppID: authorized and partner-assigned: yes/no (value withheld)   3133700 not used: confirmed
Overall: PASS | PARTIAL | BLOCKED | FAIL

| # | Check | Result (PASS/FAIL/BLOCKED/NOT IMPLEMENTED) | Evidence (redacted) |
|---|---|---|---|
| 1 | Library + SteamAPI_Init | | |
| 1b | Negative controls | | |
| 2 | Rich presence visible to a friend | | |
| 3 | Auth ticket verified by Valve | NOT IMPLEMENTED unless code added | |
| 4 | Account standing from Web API | NOT IMPLEMENTED unless code added | |
| 5 | Achievements | | |
| 6 | Leaderboards | | |
| 7 | Cloud saves across two machines | | |
| 8 | Overlay | | |
| 9 | SteamPipe upload to a non-default branch | | |

Claims NOT supported by this run: <text>
```
