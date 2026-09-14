$course = Join-Path $PSScriptRoot 'Transformer_Full_Instructional_Course.md'
$slides = Join-Path $PSScriptRoot 'images'
$web = Join-Path $PSScriptRoot 'web/course.js'

$lessonCount = (Select-String -Path $course -Pattern '\(images/slide_\d+s\.jpg\)').Count
$slideCount = (Get-ChildItem -Path $slides -Filter 'slide_*.jpg').Count

if ($lessonCount -ne 30) { throw "Expected 30 lessons, got $lessonCount." }
if ($slideCount -ne 30) { throw "Expected 30 slides, got $slideCount." }
if (-not (Select-String -Path $web -Pattern 'const courseUrl' -Quiet)) { throw 'The web course loader is missing.' }

Write-Host "[OK] 30 complete lessons, 30 slides, and the web course loader are present."
