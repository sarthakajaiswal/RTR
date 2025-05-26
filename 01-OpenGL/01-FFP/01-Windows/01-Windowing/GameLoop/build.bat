cl.exe /c /EHsc window.c 
rc.exe Window.rc 
link.exe Window.obj Window.res user32.lib gdi32.lib /SUBSYSTEM:WINDOWS 
