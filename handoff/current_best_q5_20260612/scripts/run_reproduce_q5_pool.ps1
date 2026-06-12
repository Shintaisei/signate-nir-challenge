Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

$RepoScript = Join-Path $RepoRoot "scripts\nir_slot1_testnear_branch_search.py"
$SnapshotScript = Resolve-Path (Join-Path $PSScriptRoot "nir_slot1_testnear_branch_search.py")

$RepoHash = (Get-FileHash -LiteralPath $RepoScript -Algorithm SHA256).Hash
$SnapshotHash = (Get-FileHash -LiteralPath $SnapshotScript -Algorithm SHA256).Hash
if ($RepoHash -ne $SnapshotHash) {
    throw "Repo script hash differs from handoff snapshot. Review the diff before reproducing q5. repo=$RepoHash snapshot=$SnapshotHash"
}

python scripts/nir_slot1_testnear_branch_search.py --focus-best --anchor-current-best --cluster-count 80
