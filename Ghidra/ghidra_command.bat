@echo off
call .\locations.bat
cd %project_location%

for /R "%malware_location%" %%F in (*) do (
    echo Processing file: %%F
    %ghidra_install_location%\support\analyzeHeadless.bat %project_location% %project_name% -import "%%F" -postScript %script_name% -deleteProject
)
