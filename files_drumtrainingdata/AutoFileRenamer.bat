@echo off
rem This script renames all files in the current folder to <scriptname>_001.ext, <scriptname>_002.ext, ...
rem I used it to clean my own drum sample database, you do not need it at all for my pytorch code to work.
rem @ hides text from console, prevents each command from being printed to the console (such as this comment)

rem creates a local scope for variables, enables use of !var! instead of %var% for modifying vars inside loops
setlocal enabledelayedexpansion

rem Get this batch file name (without extension), %~n0 = only the name part of script name (%0)
set "prefix=%~n0"
set count=1

rem rename all files to temporary names to avoid conflicts
rem for /f     = loop over text output line by line
rem "delims="  = do not split the line, keep full filename (including spaces)
rem %%f        = variable holding the current filename
rem dir        = list files
rem /b         = filenames only
rem /a-d       = attribute filter by files only (no directories)
rem ^|         = send output to next command (escaped pipe)
rem sort       = sort filenames alphabetically
for /f "delims=" %%f in ('dir /b /a-d ^| sort') do (

    rem rename everything to temp_xyz without the script itself
    rem /I     = case-insensitive comparison
    rem %%~nxf = current file name + extension
    rem %~nx0  = this script’s name + extension
    if /I not "%%~nxf"=="%~nx0" (
        ren "%%f" "tmp_%%f"
    )
)

rem Now rename tmp_ files to final numbered names
rem findstr /b "tmp_" = only lines beginning with "tmp_"
for /f "delims=" %%f in ('dir /b /a-d ^| findstr /b "tmp_" ^| sort') do (

    rem Build a zero-padded number, trimmed later
    set num=00!count!

    rem !num:~-3! = take last 3 characters only
    set num=!num:~-3!

    rem !prefix!  = script name (e.g. rename)
    ren "%%f" "!prefix!_!num!%%~xf"

    rem arithmetic operation
    set /a count+=1
)

echo Done!

rem wait for user key press so window doesn’t close
pause