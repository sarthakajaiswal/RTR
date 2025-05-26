// win32 headers 
#include <Windows.h> 

// standard headers 
#include <stdio.h> 

// custom headers 
#include "Window.h" 

// macros 
#define WIN_WIDTH   800 
#define WIN_HEIGHT  600 

// global variable declarations 
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM); 

// global variable declarations 
// variables related to full-screen 
BOOL gbFullScreen = FALSE; 
HWND ghwnd = NULL; 
DWORD dwStyle; 
WINDOWPLACEMENT wpPrev; 

// variable related with file i/o 
char gszLogFileName[] = "log.txt"; 
FILE *gpFile = NULL; 

// active window related variables 
BOOL gbActiveWindow = FALSE; 

// exit keypressed related 
BOOL gbEscapeKeyIsPressed = FALSE; 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) 
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
    
    // create a log file 
    gpFile = fopen(gszLogFileName, "w"); 
    if(gpFile == NULL) 
    {
        MessageBox(NULL, TEXT("Log file creation failed"), TEXT("Error"), MB_OK); 
        exit(0); 
    }
    else 
    {
        fprintf(gpFile, "Program started successfully\n"); 
    }

    // window class initialization 
    wndClass.cbSize = sizeof(WNDCLASSEX); 
    wndClass.cbClsExtra = 0; 
    wndClass.cbWndExtra = 0; 
    wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; 
    wndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); 
    wndClass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wndClass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wndClass.hInstance = hInstance; 
    wndClass.lpfnWndProc = WndProc; 
    wndClass.lpszClassName = szAppName; 
    wndClass.lpszMenuName = NULL; 

    // Registration of window class 
    RegisterClassEx(&wndClass); 

    // create window 
    hwnd = CreateWindowEx(
        WS_EX_APPWINDOW, 
        szAppName, 
        TEXT("Sarthak A Jaiswal"), 
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

    ghwnd = hwnd; 

    // show window 
    ShowWindow(hwnd, iCmdShow); 

    UpdateWindow(hwnd); 

    // initialize 
    int result = initialize(); 
    if(result != 0) 
    {
        fprintf(gpFile, "initialize() failed\n"); 
        DestroyWindow(hwnd); 
        hwnd = NULL; 
    }
    else 
    {
        fprintf(gpFile, "initialize() completed successfully"); 
    }

    SetForegroundWindow(hwnd); 
    SetFocus(hwnd); 

    // game loop 
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

    // ininitialize 
    uninitialize(); 

    return ((int)msg.wParam); 
}

// callback function 
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) 
{
    // function declarations 
    void toggleFullScreen(void); 
    void resize(int, int); 
    void uninitialize(void); 

    // code 
    switch(iMsg) 
    {
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
                    if(gbFullScreen == FALSE) 
                    {
                        toggleFullScreen(); 
                        gbFullScreen = TRUE; 
                    }
                    else 
                    {
                        toggleFullScreen(); 
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

        case WM_DESTROY: 
            PostQuitMessage(0); 
            break; 
        
        default: 
            break; 
    }

    return (DefWindowProc(hwnd, iMsg, wParam, lParam)); 
}

void toggleFullScreen(void) 
{
    // variable declarations 
    MONITORINFO mi; 

    // code 
    if(gbFullScreen == FALSE) 
    {
        dwStyle = GetWindowLong(ghwnd, GWL_STYLE); 
        if(dwStyle & WS_OVERLAPPEDWINDOW) 
        {
            ZeroMemory((void*)&mi, sizeof(MONITORINFO)); 
            mi.cbSize = sizeof(MONITORINFO); 

            if(GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi)) 
            {
                SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW); 
                SetWindowPos(
                    ghwnd, 
                    HWND_TOP, 
                    mi.rcMonitor.left, 
                    mi.rcMonitor.top, 
                    mi.rcMonitor.right - mi.rcMonitor.left, 
                    mi.rcMonitor.bottom - mi.rcMonitor.top, 
                    SWP_NOZORDER | SWP_FRAMECHANGED 
                ); 
            }
            ShowCursor(FALSE); 
        }
        else 
        {
            SetWindowPlacement(ghwnd, &wpPrev); 
            SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW); 
            SetWindowPos(
                ghwnd, 
                HWND_TOP, 
                0, 0, 0, 0, 
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOZORDER
            );
            
            ShowCursor(TRUE); 
        }
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
        fprintf(gpFile, "Program terminated successfully.\n"); 
        fclose(gpFile); 
        gpFile = NULL; 
    }
}

