// standard header files 
#include <Windows.h> 
#include <stdio.h> 
#include <stdlib.h> 

// openGL related header files 
#include <gl\GL.h> 

// custom header files 
#include "OGL.h" 

// standard libraries 
#pragma comment(lib, "user32.lib") 
#pragma comment(lib, "gdi32.lib") 

// openGL related libraries 
#pragma comment(lib, "openGL32.lib")

// macros 
#define WIN_WIDTH   800 
#define WIN_HEIGHT  600 

// global variable declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);  

// view port related functions 
void setViewport0(void); 
void setViewport1(void); 
void setViewport2(void); 
void setViewport3(void); 
void setViewport4(void); 
void setViewport5(void); 
void setViewport6(void); 
void setViewport7(void); 
void setViewport8(void); 
void setViewport9(void); 

// global variable declarations 
// variables related to full-screen 
BOOL gbFullScreen = FALSE; 
HWND ghwnd = NULL; 
DWORD dwStyle; 
WINDOWPLACEMENT wpPrev; 

// variable related to file-IO 
char gszLogFileName[] = "log.txt"; 
FILE *gpFile = NULL; 

// active Window related variables 
BOOL gbActiveWindow = FALSE; 

// exit key pressed related 
BOOL gbEscapeKeyIsPressed; 

// OpenGL related global variables 
HDC ghdc = NULL; 
HGLRC ghrc = NULL; 

// view-port related variables 
RECT clientRect;    // for storing client area sdimentions  
GLint viewportOriginX = 0, viewportOriginY = 0; 
GLint viewportWidth, viewportHeight; 

void (*pfnSetViewport)(void) = NULL; 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) 
{
    // local function declarations 
    int initialize(void); 
    void display(void); 
    void update(void); 
    void uninitialize(void); 

    // variable declarations 
    static TCHAR szClassName[] = TEXT("The Standard Window"); 

    MSG msg; 
    WNDCLASSEX wnd; 
    HWND hwnd; 

    BOOL bDone = FALSE; 

    // code 
    // create log file 
    gpFile = fopen(gszLogFileName, "w"); 
    if(gpFile == NULL) 
    {
        MessageBox(NULL, TEXT("Log file creation failed"), TEXT("File Error"), MB_OK); 
        exit(0); 
    }
    else 
    {
        fprintf(gpFile, "Program started Successfully"); 
    }

    ZeroMemory((void*)&wnd, sizeof(WNDCLASSEX)); 
    ZeroMemory((void*)&msg, sizeof(MSG)); 

    // window class initialization
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wnd.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH); 
    wnd.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wnd.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wnd.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wnd.lpfnWndProc = WndProc; 
    wnd.hInstance = hInstance; 
    wnd.lpszClassName = szClassName; 
    wnd.lpszMenuName = NULL; 

    // window class registration 
    RegisterClassEx(&wnd); 

    pfnSetViewport = setViewport0; 

    // create window 
    hwnd = CreateWindowEx(
                WS_EX_APPWINDOW, 
                szClassName, 
                TEXT("Sarthak Ayodhyaprasad Jaiswal"), 
                WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS, 
                (GetSystemMetrics(SM_CXSCREEN)/2 - WIN_WIDTH/2), 
                (GetSystemMetrics(SM_CYSCREEN)/2 - WIN_HEIGHT/2), 
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

    // update window 
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
        fprintf(gpFile, "initialize() completed successfully\n"); 
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

    // uninitialize 
    uninitialize(); 

    return ((int)msg.wParam); 
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) 
{
    // local function declarations 
    void toggleFullScreen(void); 
    void resize(void); 
    void uninitialize(void); 

    // code 
    switch(uMsg) 
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

        case WM_ERASEBKGND: 
            return (0); 

        case WM_SIZE: 
            viewportHeight = LOWORD(lParam); 
            viewportWidth = HIWORD(lParam); 
            resize(); 
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
                case 'f': 
                case 'F': 
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

                case '0': 
                    pfnSetViewport = setViewport0; 
                    pfnSetViewport(); 
                    break; 

                case '1': 
                    pfnSetViewport = setViewport1; 
		    pfnSetViewport(); 
                    break; 

                case '2': 
                    pfnSetViewport = setViewport2; 
                    pfnSetViewport(); 
                    break; 

                case '3': 
                    pfnSetViewport = setViewport3; 
                    pfnSetViewport(); 
                    break; 
                
                case '4': 
                    pfnSetViewport = setViewport4; 
                    pfnSetViewport();     
                    break; 

                case '5': 
                    pfnSetViewport = setViewport5; 
                    pfnSetViewport(); 
                    break; 
                
                case '6': 
                    pfnSetViewport = setViewport6; 
                    pfnSetViewport(); 
                    break; 

                case '7': 
                    pfnSetViewport = setViewport7; 
                    pfnSetViewport(); 
                    break; 

                case '8': 
                    pfnSetViewport = setViewport8; 
                    pfnSetViewport(); 
                    break; 

                case '9': 
                    pfnSetViewport = setViewport9; 
                    pfnSetViewport(); 
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

    return (DefWindowProc(hwnd, uMsg, wParam, lParam)); 
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
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED
        ); 
        ShowCursor(TRUE); 
    }
}

int initialize(void) 
{
    PIXELFORMATDESCRIPTOR pfd; 
    int iPixelFormatIndex; 

    // code 
    ZeroMemory((void*)&pfd, sizeof(PIXELFORMATDESCRIPTOR)); 
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR); 
    pfd.nVersion = 1; 
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER; 
    pfd.iPixelType = PFD_TYPE_RGBA; 
    pfd.cColorBits = 32; 
    pfd.cRedBits = 8; 
    pfd.cGreenBits = 8;  
    pfd.cBlueBits = 8; 
    pfd.cAlphaBits = 8; 

    ghdc = GetDC(ghwnd); 
    if(ghdc == NULL) 
    {
        fprintf(gpFile, "GetDC() failed\n"); 
        return (-1); 
    }

    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd); 
    if(iPixelFormatIndex == 0) 
    {
        fprintf(gpFile, "ChoosePixelFormat() failed\n"); 
        return (-2); 
    }

    if(SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE) 
    {
        fprintf(gpFile, "iPixelFormatIndex() failed\n"); 
        return (-3); 
    }

    ghrc = wglCreateContext(ghdc); 
    if(ghrc == NULL) 
    {
        fprintf(gpFile, "wglCreateContext() failed\n"); 
        return (-4); 
    }

    if(wglMakeCurrent(ghdc, ghrc) == FALSE) 
    {
        fprintf(gpFile, "wglMakeCurrent() failed\n"); 
        return (-5); 
    }

    // ********** FROM HERE OPENGL CODE STARTS ************ 
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 

    return (0); 
}

void resize(void) 
{
    if(viewportHeight <= 0) 
    {
        viewportHeight = 1; 
    }

    pfnSetViewport(); 
}

void display(void) 
{

    // code 
    glClear(GL_COLOR_BUFFER_BIT); 

    glBegin(GL_TRIANGLES);     
    glVertex2f(0.0f, 1.0f); 
    glVertex2f(-1.0f, -1.0f); 
    glVertex2f(1.0f, -1.0f); 
    glEnd(); 

    SwapBuffers(ghdc); 
}

void update(void) 
{
    //code 
}

void uninitialize(void) 
{
    // function declarations 
    void toggleFullScreen(void); 

    // code 
    if(gbFullScreen == TRUE) 
    {
        toggleFullScreen(); 
        gbFullScreen = FALSE; 
    }

    if(wglGetCurrentContext() == ghrc) 
    {
        wglMakeCurrent(NULL, NULL); 
    }

    if(ghrc) 
    {
        wglDeleteContext(ghrc); 
        ghrc = NULL; 
    }

    if(ghdc) 
    {
        ReleaseDC(ghwnd, ghdc); 
        ghdc = NULL; 
    }

    if(ghwnd) 
    {
        DestroyWindow(ghwnd); 
        ghwnd = NULL; 
    }

    if(gpFile) 
    {
        fprintf(gpFile, "Program Terminated Successfully"); 
        fclose(gpFile); 
        gpFile = NULL; 
    }
}

void setViewport0(void) 
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = 0; 
    glViewport(viewportOriginX, viewportOriginY, clientRect.right, clientRect.bottom); 
}

void setViewport1(void) 
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = 0; 
    glViewport(viewportOriginX, viewportOriginY, clientRect.right/2, clientRect.bottom/2); 
}

void setViewport2(void) 
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = clientRect.right/2; 
    viewportOriginY = 0; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right/2, 
        clientRect.bottom/2
    ); 
}

void setViewport3(void) 
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = clientRect.right/2; 
    viewportOriginY = clientRect.bottom/2; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right/2, 
        clientRect.bottom/2
    );     
}

void setViewport4(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = clientRect.bottom/2; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right/2, 
        clientRect.bottom/2
    );     
}

void setViewport5(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = 0; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right/2, 
        clientRect.bottom
    ); 
}

void setViewport6(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = clientRect.right/2; 
    viewportOriginY = 0; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right/2, 
        clientRect.bottom
    ); 
}

void setViewport7(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = clientRect.bottom/2; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right, 
        clientRect.bottom/2 
    ); 
}

void setViewport8(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = 0; 
    viewportOriginY = 0; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        clientRect.right, 
        clientRect.bottom/2
    ); 
}

void setViewport9(void)
{
    GetClientRect(ghwnd, &clientRect); 
    viewportOriginX = clientRect.right/5; 
    viewportOriginY = clientRect.bottom/5; 
    glViewport(
        viewportOriginX, 
        viewportOriginY, 
        3 * clientRect.right / 5, 
        3 * clientRect.bottom / 5
    ); 
}
