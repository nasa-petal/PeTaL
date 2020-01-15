docker build . -t petal_img
START docker run --name petal --rm -it -p 5000:5000 petal_img

@echo off && setlocal enabledelayedexpansion
:loop
timeout -t 1 >nul  
for /R . %%f in (*) do call :check %%f
goto :loop

:check
set B=%1
echo %1|find /V ".git">nul || goto :EOF
echo %1|find /V ".idea">nul || goto :EOF
echo %~a1|find "a">nul || goto :EOF
echo docker cp !B:%CD%\=! petal:/petal/!B:%CD%\=!
docker cp !B:%CD%\=! petal:/petal/!B:%CD%\=!
attrib -a %1
    