# Steam integration status

Checked on 2026-10-18 against Steam's public store and the repository's current implementation.

## App ID correction

Steam App ID `3133700`, previously advertised here for HiveMind, belongs to
[Fantasy Grounds - Raiders of the Ghost Zone](https://store.steampowered.com/app/3133700/),
a paid DLC published by SmiteWorks. This is also reported by
[Steam's app-details API](https://store.steampowered.com/api/appdetails?appids=3133700).
It is not a HiveMind installation link. A verified HiveMind store listing and
Valve-assigned App ID have not been provided in this repository.

## What runs locally

- The launchers in `steam_dist/HiveMind_Steam_FreeEdition` run Python modules from
  the repository root. The folder requires a configured Python environment and
  the rest of the repository; it is not a standalone Steam installer.
- On `main`, `SteamworksSDKBridge` generates HMAC receipts locally and stores
  leaderboard entries in a Python dictionary. These are simulated Steam features,
  not authenticated Steam sessions or uploads to Valve's leaderboards.
- The package builder writes local JSON files for the manifest and cloud-save
  profiles. Writing `release_state: released` or a SteamPipe build script does
  not establish a published Steam release or sync data with Steam Cloud.
- The domain swarm demo runs its participants and reward ledger locally. Trying
  it does not, by itself, connect a volunteer GPU to a public training network.

See the [Steam bridge](../src/hivemind/client/steamworks.py),
[package builder](../src/hivemind/game/steam_edition.py), and
[swarm demo](../src/hivemind/cli/demo_hf_domain_swarm.py).

## Trying the prototype

Follow the [development setup](../CONTRIBUTING.md#3-local-development-setup-8-gb-gpu-or-cpu),
then run `hivemind play-hack-game` or `hivemind serve-hack-hud` from the repository
root. Opening `web_ui/index.html` directly only inspects the offline HUD and its
simulations; it does not start GPU training.

The existing App ID, Depot ID and release metadata in code and generated build
files are legacy demo values, not verified HiveMind publishing credentials. Do
not use them to publish or authenticate as HiveMind. A future Steam release needs
the maintainer's actual assigned identifiers, native integration and a verified
store link. This documentation correction does not assign replacement IDs or
change the demo's runtime interfaces.
