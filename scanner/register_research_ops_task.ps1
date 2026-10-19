param(
    [string]$TaskName = "Kronos Daily Research Ops",
    [string]$StartTime = "13:30"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $PSScriptRoot "run_research_ops_scheduled.bat"

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Scheduled research runner not found: $runner"
}

$action = New-ScheduledTaskAction `
    -Execute $runner `
    -WorkingDirectory $repoRoot

$startAt = [datetime]::Today.Add([timespan]::Parse($StartTime))
if ((Get-Date) -gt $startAt.AddMinutes(60)) {
    # Registering after the recovery window should not manufacture missed runs
    # or immediately launch a degraded after-hours catch-up cycle.
    $startAt = $startAt.AddDays(1)
}
$triggers = @(
    New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
        -At $startAt
    New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
        -At ($startAt.AddMinutes(30))
    New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
        -At ($startAt.AddMinutes(60))
)

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $triggers `
    -Settings $settings `
    -Description "Runs the Kronos evidence cycle every market weekday with idempotent same-session recovery launches." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object -ExpandProperty Settings
