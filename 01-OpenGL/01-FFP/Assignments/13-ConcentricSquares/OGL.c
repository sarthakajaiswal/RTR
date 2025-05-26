// standard header files 
#include <Windows.h> 
#include <stdio.h> 
#include <stdlib.h> 

// openGL related header files 
#include <gl\GL.h> 

// standard libraries 
#pragma comment(lib, "user32.lib") 
#pragma comment(lib, "gdi32.lib") 

// openGL related libraries 
#pragma comment(lib, "openGL32.lib")

// macros 
#define WIN_WIDTH   800 
#define WIN_HEIGHT  600 

// structure definition 
struct Color 
{
    float r, g, b, a; 
}; 

// global variable declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);  

// global variable declarations 
// variables related to full-screen 
BOOL gbFullScreen = FALSE; 
HWND ghwnd = NULL; 
DWORD dwStyle; 
WINDOWPLACEMENT wpPrev; 

// active Window related variables 
BOOL gbActiveWindow = FALSE; 

// exit key pressed related 
BOOL gbEscapeKeyIsPressed = FALSE; 

// OpenGL related global variables 
HDC ghdc = NULL; 
HGLRC ghrc = NULL; 

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
    ZeroMemory((void*)&wnd, sizeof(WNDCLASSEX)); 
    ZeroMemory((void*)&msg, sizeof(MSG)); 

    // window class initialization
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wnd.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH); 
    wnd.hIcon = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hIconSm = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wnd.lpfnWndProc = WndProc; 
    wnd.hInstance = hInstance; 
    wnd.lpszClassName = szClassName; 
    wnd.lpszMenuName = NULL; 

    // window class registration 
    RegisterClassEx(&wnd); 

    // create window 
    hwnd = CreateWindowEx(
                WS_EX_APPWINDOW, 
                szClassName, 
                TEXT("Sarthak Jaiswal"), 
                WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS, 
                CW_USEDEFAULT, 
                CW_USEDEFAULT, 
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
        DestroyWindow(hwnd); 
        hwnd = NULL; 
        return (0); 
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
    void resize(int, int); 
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
        return (-1); 
    }

    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd); 
    if(iPixelFormatIndex == 0) 
    {
        return (-2); 
    }

    if(SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE) 
    {
        return (-3); 
    }

    ghrc = wglCreateContext(ghdc); 
    if(ghrc == NULL) 
    {
        return (-4); 
    }

    if(wglMakeCurrent(ghdc, ghrc) == FALSE) 
    {
        return (-5); 
    }

    // ********** FROM HERE OPENGL CODE STARTS ************ 
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 

    return (0); 
}

void resize(int width, int height) 
{
    if(height <= 0) 
    {
        height = 1; 
    }

    glViewport(0, 0, (GLsizei)width, (GLsizei)height); 
}

void display(void) 
{
    // variable declarations 
    static const numberOfSquares = 10; 
    float coordinateOfVertexOfSquare; /* co-ordinate of vertex of a triangle 
                                            this is used as x as well as y co-ordinate to get equilateral triangle 
                                        */
    float deltaSize; /* size difference between two consecutive triangles */ 

    struct Color squareColors[10] = {
                                        {1.0f, 0.0f, 0.0f, 1.0f}, 
                                        {0.0f, 1.0f, 0.0f, 1.0f}, 
                                        {0.0f, 0.0f, 1.0f, 1.0f}, 
                                        {1.0f, 1.0f, 0.0f, 1.0f}, 
                                        {1.0f, 0.0f, 1.0f, 1.0f}, 
                                        {0.0f, 1.0f, 1.0f, 1.0f}, 
                                        {1.0f, 1.0f, 1.0f, 1.0f}, 
                                        {1.0f, 0.5f, 0.0f, 1.0f}, 
                                        {0.5f, 0.5f, 0.5f, 1.0f}, 
                                        {0.0f, 0.5f, 1.0f, 1.0f} 
                                    }; 
    // code 
    glClear(GL_COLOR_BUFFER_BIT); 

    coordinateOfVertexOfSquare = 0.05f; 
    deltaSize = (0.9f - coordinateOfVertexOfSquare) / numberOfSquares; 

    for(int i = 0; i < numberOfSquares; ++i) 
    {
        glBegin(GL_LINE_LOOP); 
        glColor3f(squareColors[i].r, squareColors[i].g, squareColors[i].b); 
        glVertex3f(coordinateOfVertexOfSquare, coordinateOfVertexOfSquare, 0.0f); 
        glVertex3f(-coordinateOfVertexOfSquare, coordinateOfVertexOfSquare, 0.0f); 
        glVertex3f(-coordinateOfVertexOfSquare, -coordinateOfVertexOfSquare, 0.0f); 
        glVertex3f(coordinateOfVertexOfSquare, -coordinateOfVertexOfSquare, 0.0f); 
        glEnd(); 

        coordinateOfVertexOfSquare += deltaSize; 
    } 

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
}

