
----------------------------------------------------------------------------
    Sir must have answered this, but I am not remebering the lecture date.. 
    Here if initialize() fails why dont we return as window is destroyed  
----------------------------------------------------------------------------

int result = initialize(); 
if(result != 0) 
{
    fprintf(gpFile, "initialize() failed"); 
    DestroyWindow(hwnd); 
    hwnd = NULL; 
}
else
{
    fprintf(gpFile, "initialize() successfull"); 
}

SetForegroundWindow(); 
SetFocus(); 
