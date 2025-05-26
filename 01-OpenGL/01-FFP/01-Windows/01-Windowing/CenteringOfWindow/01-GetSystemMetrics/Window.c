// win32 headers 
#include <Windows.h> 
#include <stdio.h> 
#include <stdlib.h> 

#include "Window.h"

// Macros 
#define WIN_WIDTH   800 
#define WIN_HEIGHT  600 

// global function declarations 
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM); 

// global variable declarations 
// variables related with full-screen  
BOOL gbFullScreen = FALSE; 
HWND ghWnd = NULL; 
DWORD dwStyle; 
WINDOWPLACEMENT wpPrev; 

// variables related with file I/O 
char gszLogFileName[] = "Log.txt"; 
FILE* gpFile = NULL; 

// active window related variable 
BOOL gbActiveWindow = FALSE; 

// exit key pressed related 
BOOL gbEscapeKeyIsPressed = FALSE; 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iShowCmd) 
{
    // local function declarations 
    int initialize(void); 
    void display(void); 
    void update(void); 
    void uninitialize(void); 

    // variable declarations 
    WNDCLASSEX wndClass; 
    HWND hwnd; 
    MSG msg; 
    TCHAR szAppName[] = TEXT("RTR 6.0"); 
    BOOL bDone = FALSE; 

    // code 

    // create log file 
    gpFile = fopen(gszLogFileName, "w"); 
    
    if(gpFile == NULL) 
    {
        MessageBox(NULL, TEXT("Logfile creation failed"), TEXT("Error"), MB_OK); 
        exit(0);  
    }
    else 
    {
        fprintf(gpFile, "Program started successfully\n"); 
    }

    // window class initialization
    wndClass.cbSize = sizeof(WNDCLASSEX); 
    wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; 
    wndClass.cbClsExtra = 0; 
    wndClass.cbWndExtra = 0; 
    wndClass.lpfnWndProc = WndProc; 
    wndClass.hInstance = hInstance; 
    wndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); 
    wndClass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wndClass.lpszClassName = szAppName; 
    wndClass.lpszMenuName = NULL; 
    wndClass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 

    // registration of window class 
    RegisterClassEx(&wndClass); 

    // create window 
    hwnd = CreateWindowEx(
                        WS_EX_APPWINDOW, 
                        szAppName, 
                        TEXT("Sarthak Jaiswal"), 
                        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 
                        GetSystemMetrics(SM_CXSCREEN)/2 - WIN_WIDTH/2, 
                        GetSystemMetrics(SM_CYSCREEN)/2 - WIN_HEIGHT/2, 
                        WIN_WIDTH, 
                        WIN_HEIGHT, 
                        NULL, 
                        NULL, 
                        hInstance, 
                        NULL
                    ); 
    ghWnd = hwnd; 
    
    // show window 
    ShowWindow(hwnd, iShowCmd); 

    // paint background of window 
    UpdateWindow(hwnd); 

    // initialize 
    int result = initialize(); 

    if(result != 0) 
    {
        fprintf(gpFile, "Initilize() failed\n"); 
        DestroyWindow(hwnd); 
        hwnd = NULL; 
    }
    else 
    {
        fprintf(gpFile, "initialize() completed successfully\n"); 
    }

    // set this window as foreground and active window 
    SetForegroundWindow(hwnd); 
    SetFocus(hwnd); 

    // Game Loop 
    while(bDone == FALSE) 
    {
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) 
        {
            if(msg.message == WM_QUIT) 
            {
                bDone = TRUE; 
            }
            else 
            {
                TranslateMessage(&msg); 
                DispatchMessage(&msg); 
            }
        }
        else 
        {
            if(gbActiveWindow == TRUE) 
            {
                if(gbEscapeKeyIsPressed == TRUE) 
                {
                    bDone = TRUE; 
                }

                // render 
                display(); 

                // update 
                update();
            }
        }
    }

    // uninitialize 
    uninitialize(); 

    return ((int)msg.wParam); 
} 

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // local function declarations 
    void ToggleFullScreen(void); 
    void resize(int, int); 
    void uninitialize(void); 

    switch(iMsg) 
    {
        case WM_DESTROY : 
            PostQuitMessage(0); 
            break; 

        case WM_CREATE: 
            ZeroMemory((void*)&wpPrev, sizeof(WINDOWPLACEMENT)); 
            wpPrev.length = sizeof(WINDOWPLACEMENT); 
            break; 

        case WM_SETFOCUS: 
            gbActiveWindow = TRUE; 
            break; 

        case WM_KILLFOCUS: 
            gbActiveWindow = FALSE; 
            break; 

        case WM_SIZE: 
            resize(LOWORD(lParam), HIWORD(lParam)); 
            break; 

        case WM_KEYDOWN: 
            switch(wParam) 
            {
                case VK_ESCAPE: 
                    gbEscapeKeyIsPressed = TRUE; 
                    break; 
                
                default: 
                    break; 
            }
            break; 

        case WM_CHAR: 
            switch(wParam) 
            {
                case 'F': 
                case 'f': 
                    if (gbFullScreen == FALSE) 
                    {
                        // dwStyle = GetWindowLong(); 
                        ToggleFullScreen(); 
                        gbFullScreen = TRUE; 
                    }
                    else 
                    {
                        ToggleFullScreen(); 
                        gbFullScreen = FALSE; 
                    }
                    break; 

                default: 
                    break; 
            } 
            break;

        case WM_CLOSE: 
            uninitialize(); 
            break; 
    }

    return (DefWindowProc(hwnd, iMsg, wParam, lParam)); 
}

void ToggleFullScreen(void) 
{
    // variable declarations 
    MONITORINFO mi; 

    // code 
    if(gbFullScreen == FALSE) 
    {
        dwStyle = GetWindowLong(ghWnd, GWL_STYLE); 
        if(dwStyle & WS_OVERLAPPEDWINDOW) 
        {
            ZeroMemory((void*)&mi, sizeof(MONITORINFO)); 
            mi.cbSize = sizeof(MONITORINFO); 
            if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi)) 
            {
                SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW); 
                SetWindowPos(ghWnd, HWND_TOP, 
                            mi.rcMonitor.left, 
                            mi.rcMonitor.top, 
                            mi.rcMonitor.right-mi.rcMonitor.left, 
                            mi.rcMonitor.bottom-mi.rcMonitor.top, 
                            SWP_NOZORDER | SWP_FRAMECHANGED
                        ); 
            }
        }
        ShowCursor(FALSE); 
    }
    else 
    {
        SetWindowPlacement(ghWnd, &wpPrev); 
        SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW); 
        SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED); 
        ShowCursor(TRUE); 
    }
}

int initialize(void) 
{
    // code 
    return (0); 
}

void resize(int width, int height) 
{
    // code 
}

void display(void) 
{
    // code 
}

void update(void) 
{
    // code 
}

void uninitialize(void) 
{
    // code 
    // close the file 
    if(gpFile) 
    {
        fprintf(gpFile, "Program terminated successfully"); 
        fclose(gpFile); 
        gpFile = NULL; 
    } 
}
