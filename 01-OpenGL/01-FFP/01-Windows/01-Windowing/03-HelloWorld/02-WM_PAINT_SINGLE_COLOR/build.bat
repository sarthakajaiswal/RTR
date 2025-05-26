del Window.exe 
cl.exe /c /EHsc Window.c 
link.exe gdi32.lib user32.lib Window.obj 
del *.obj 