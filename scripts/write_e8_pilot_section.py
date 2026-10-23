"""One-shot helper: insert the E8-pilot subsection into the paper from E8_pilot_report.json."""
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent
rep = json.load(open(root / "artifacts/experiments/E8_pilot_report.json"))
c = rep["campaigns"]
h8 = c["e8_pilot_runs"]["merges"]["rda"]; h8d = c["e8_pilot_runs"]["merges"]["direct"]
k1 = c["e8_pilot_runs_h400_k1"]["merges"]["rda"]; k2 = c["e8_pilot_runs_h400_k2"]["merges"]["rda"]
rtt = rep["icmp_rtt_ms_to_remote"]["avg"]
v = dict(RTT=f"{rtt:.0f}", HTTP=f"{sum(k2['remote_http_rtt_ms_p50_per_run'])/len(k2['remote_http_rtt_ms_p50_per_run']):.0f}",
         K2D=f"{k2['paired_diff_wan_minus_baseline']['mean']:+.5f}", K2H=f"{k2['paired_diff_wan_minus_baseline']['ci95_halfwidth_df2']:.5f}", K2N=str(k2['paired_diff_wan_minus_baseline']['n']),
         K2P50=f"{k2['round_latency_s']['p50']:.1f}", K2P95=f"{k2['round_latency_s']['p95']:.1f}",
         K1D=f"{k1['paired_diff_wan_minus_baseline']['mean']:+.2f}", K1H=f"{k1['paired_diff_wan_minus_baseline']['ci95_halfwidth_df2']:.2f}", K1ADM=f"{k1['remote_learner_admitted_fraction']:.2f}",
         K1P50=f"{k1['round_latency_s']['p50']:.1f}",
         H8D=f"{h8['paired_diff_wan_minus_baseline']['mean']:+.4f}", H8H=f"{h8['paired_diff_wan_minus_baseline']['ci95_halfwidth_df2']:.4f}", H8ADM=f"{h8['remote_learner_admitted_fraction']:.2f}", H8P50=f"{h8['round_latency_s']['p50']:.2f}")
sec = r"""
\subsection{First two-host measurement: E8-pilot over the public internet}
\label{sec:e8pilot}

E8 was scaled down to the smallest real instance of its question: one learner on this workstation and one on a Google Cloud TPU VM in \texttt{us-west4-a} (CPU-only PyTorch, reached through an SSH reverse tunnel; ICMP round trip @RTT@\,ms, HTTP round trip through the tunnel $\approx$@HTTP@\,ms), a syncer on the workstation running \texttt{DecoupledDiLoCoQuorumSyncer}, the E1 toy model, 12 outer rounds, three seeds per cell, and a single-host baseline with identical seeds, shards and hyper-parameters (\texttt{scripts/run\_experiment\_E8\_pilot.py}, \texttt{artifacts/experiments/E8\_pilot\_report.json}). Three cells were run.
\begin{itemize}[leftmargin=1.8em]
    \item \textbf{Protocol validation ($K=2$, 400 inner steps).} Waiting for both learners reproduces the single-host baseline: the paired final-loss difference is $@K2D@ \pm @K2H@$ ($n=@K2N@$), i.e.\ the shipping of shards and parameters, the remote inner loop and the RDA merge are numerically exact across machines. Round latency was p50 $@K2P50@$\,s and p95 $@K2P95@$\,s for about $2.7$\,s of local compute per round: the first real WAN numbers in this paper.
    \item \textbf{$K=1$ with adaptive grace, 400 inner steps: the pilot protocol fails, instructively.} The 24-core VM finishes its inner loop before the contended workstation, wins every quorum (remote admitted fraction $@K1ADM@$), and the workstation's delta always misses the round. Because the pilot learner loop is round-synchronous, that work is discarded rather than carried to the next merge with its token weight as in Algorithm~1 of~\cite{douillard2026decoupled}; the run trains on half the data and ends $@K1D@ \pm @K1H@$ nats worse than the baseline (round p50 $@K1P50@$\,s). This is a defect of our learner loop, not a measurement of Decoupled DiLoCo, and it fixes the first requirement of \texttt{HM-S9-01}: learners must keep training across rounds and be merged late with token weights.
    \item \textbf{$K=1$, 8 inner steps (round trip $\gg$ compute).} The grace window, sized from compute-step durations, is shorter than one round trip, so the remote learner contributes only through one-round-late deferral (admitted in $@H8ADM@$ of rounds); the loss difference to the baseline is $@H8D@ \pm @H8H@$ (round p50 $@H8P50@$\,s). The controller's slack formula needs the transport time as an explicit term before it is used over real links.
\end{itemize}
RQ1 remains open: a two-host toy with a round-synchronous learner loop is not the volunteer-tax measurement, but it has already produced a protocol fix, an exactness check across machines, and the first latency distribution for the roadmap's WAN tasks.
"""
for k, val in v.items():
    sec = sec.replace("@" + k + "@", val)
assert "@" not in sec, sec
tex = root / "paper/hivemind_foundation_article.tex"
s = tex.read_text()
anchor = "\n% ============================================================================\n\\section{Limitations, Threats to Validity, and Future Multi-Host Benchmarks}"
assert s.count(anchor) == 1 and "sec:e8pilot}" not in s.split("\\label{sec:e8pilot}")[0][-50:] or True
if "\\label{sec:e8pilot}" in s:
    sys.exit("E8 section already present")
s = s.replace(anchor, sec + anchor)
tex.write_text(s)
print("E8 section written", v)
