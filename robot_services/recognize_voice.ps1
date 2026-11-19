param([switch]$Check)
$ErrorActionPreference = 'Stop'
$recognizer = $null
try {
    Add-Type -AssemblyName System.Speech
    $installed = @([System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
        Where-Object { $_.Culture.TwoLetterISOLanguageName -eq 'en' })
    if ($installed.Count -eq 0) {
        throw 'Install a Windows English desktop speech recognizer to use voice commands.'
    }
    if ($Check) {
        @{ available = $true; culture = $installed[0].Culture.Name } | ConvertTo-Json -Compress
        exit 0
    }
    $recognizer = [System.Speech.Recognition.SpeechRecognitionEngine]::new($installed[0])
    $choices = [System.Speech.Recognition.Choices]::new()
    $choices.Add([string[]]@('walk forward', 'walk backward', 'turn left', 'turn right',
                           'stand up', 'sit down', 'stop'))
    $builder = [System.Speech.Recognition.GrammarBuilder]::new()
    $builder.Culture = $installed[0].Culture
    $builder.Append($choices)
    $recognizer.LoadGrammar([System.Speech.Recognition.Grammar]::new($builder))
    $recognizer.SetInputToDefaultAudioDevice()
    @{ ready = $true } | ConvertTo-Json -Compress
    $result = $recognizer.Recognize([TimeSpan]::FromSeconds(5))
    if ($null -eq $result) {
        @{ text = ''; confidence = 0.0 } | ConvertTo-Json -Compress
    } else {
        @{ text = $result.Text; confidence = [double]$result.Confidence } | ConvertTo-Json -Compress
    }
} catch {
    @{ error = $_.Exception.Message } | ConvertTo-Json -Compress
    exit 1
} finally {
    if ($null -ne $recognizer) { $recognizer.Dispose() }
}
