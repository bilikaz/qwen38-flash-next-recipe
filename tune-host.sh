#!/usr/bin/env bash
# tune-host.sh — the ONE host setting this kit recommends but never applies on its own (it needs root):
#
#   vm.compaction_proactiveness = 0      persisted in /etc/sysctl.d/99-myllmbox-compaction.conf
#
# WHY: the kernel's background page compactor migrates pages to build large contiguous blocks. On a DGX Spark the
# GPU's memory IS ordinary system pages, so every migrated page is first unmapped from the GPU. On a serve pinned
# close to the memory edge (this one) that measured as a 4–5 s slowdown every ~37 s — about 10 % of throughput and a
# 30 % drop in the worst 10-second window. A serving box allocates once at boot; it gains nothing from the upkeep.
# Direct compaction on a real allocation failure still works — only the proactive background pass is disabled.
#
# Shows exactly what it will run, asks once, then lets sudo prompt the normal way (the password goes into sudo's own
# prompt — never read, stored or passed by this script). Reversible:
#   sudo sysctl -w vm.compaction_proactiveness=20 && sudo rm /etc/sysctl.d/99-myllmbox-compaction.conf
set -euo pipefail
CMD='printf "vm.compaction_proactiveness = 0\n" > /etc/sysctl.d/99-myllmbox-compaction.conf && sysctl -q vm.compaction_proactiveness=0 && echo "  ✓ vm.compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness) (persisted)"'
now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
echo "current: vm.compaction_proactiveness=${now}   (want 0)"
[ "$now" = 0 ] && { echo "✓ already set — nothing to do"; exit 0; }
echo; echo "This will run AS ROOT:"; echo "  sudo bash -c '$CMD'"; echo
read -rp "Proceed? [y/N] " ans; [[ "${ans:-N}" =~ ^[Yy]$ ]] || { echo "aborted — nothing changed"; exit 0; }
sudo bash -c "$CMD"
echo "done. It is live now (no restart) and survives reboots."
