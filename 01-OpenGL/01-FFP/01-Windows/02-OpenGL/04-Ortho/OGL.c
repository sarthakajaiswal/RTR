// standard header files 
#include <Windows.h> 
#include <stdio.h>      // fopen(), fwrite() and fclose() 
#include <stdlib.h>     // exit() 

// openGL related files
#include <GL/gl.h>  

// custom header files 
#include "OGL.h" 

// OpenGL related libraries 
#pragma comment(lib, "opengl32.lib")  

// macros 
#define WIN_WIDTH 800 
#define WIN_HEIGHT 600

// global function declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

// global variable declarations 
// variables related to full screen 
BOOL gbFullScreen = FALSE; 
HWND ghwnd = NULL; 
DWORD dwStyle;  
WINDOWPLACEMENT wpPrev; 

// variables related to file-IO
char gszLogFileName[] = "log.txt"; 
FILE *gpFile = NULL; 

// active window related variable 
BOOL gbActiveWindow = FALSE; 

// exit key pressed related 
BOOL gbEscapeKeyIsPressed = FALSE; 

// OpenGL related global variables 
HDC ghdc = NULL; 
HGLRC ghrc = NULL; // handle to Graphics Library Rendering context 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nShowCmd) 
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
        MessageBox(NULL, TEXT("Log File creation failed"), TEXT("File Error"), MB_OK); 
        exit(0);    
    }
    else 
    {
        fprintf(gpFile, "Program started Successfully\n"); 
    } 

    ZeroMemory(&msg, sizeof(MSG)); 
    ZeroMemory(&wnd, sizeof(WNDCLASSEX)); 

    // window class initialization 
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wnd.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;      //cs - class style // cs_owndc - os la sangne asa DC tayar kr jo in between tu move krnaar nahis, fix memory la thevshil   
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

    // create window from registered class 
    hwnd = CreateWindowEx(
        WS_EX_APPWINDOW, 
        szClassName, 
        TEXT("Sarthak Jaiswal"), 
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 
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
    ShowWindow(hwnd, nShowCmd); 

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
        if(PeekMessage(&msg, NULL, 0, 0,PM_REMOVE)) 
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
            if(gbActiveWindow == TRUE)  // if the current window is active them only render() and display() 
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
    void toggleFullScreen(void); 
    void resize(int, int); 
    void uninitialize(void); 

    // code 
    switch(iMsg) 
    {
        case WM_CREATE: 
            ZeroMemory((void*)&wpPrev, sizeof(WINDOWPLACEMENT)); // length member in wpPrev must be set before it is used 
                                                                // this setting is to be done once at starting 
                                                                // so initializing it in WWM_CREATE.  
            wpPrev.length = sizeof(WINDOWPLACEMENT); 
            break; 
        
        case WM_SETFOCUS: 
            gbActiveWindow = TRUE; 
            break; 

        case WM_KILLFOCUS: 
            gbActiveWindow = FALSE; 
            break; 

        case WM_ERASEBKGND: 
            return(0); 

        case WM_SIZE:
            resize(LOWORD(lParam), HIWORD(lParam)); 
            break; 

        case WM_KEYDOWN: 
            switch (wParam) 
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

void toggleFullScreen(void) // camel notation 
{
    // variable declarations 
    MONITORINFO mi; 

    // code 
    if(gbFullScreen == FALSE) 
    {
        // 1) get current window style 
        dwStyle = GetWindowLong(ghwnd, GWL_STYLE); 

        // 2) Check wether window style CONTAINS ws_overlappedwindow or not  
        if(dwStyle & WS_OVERLAPPEDWINDOW)  
        {
            ZeroMemory((void*)&mi, sizeof(MONITORINFO)); 
            mi.cbSize = sizeof(MONITORINFO); 

            if(GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))  // F-flag 
            {
                // removes style 
                SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW); // go inside dwStyle -> dwStyle& 
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

        // optional 
        ShowCursor(FALSE); 
    } 
    else 
    {
        SetWindowPlacement(ghwnd, &wpPrev); 
        SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW); 
        SetWindowPos(
            ghwnd, 
            HWND_TOP, 
            0, 
            0, 
            0, 
            0, 
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED 
        ); 

        ShowCursor(TRUE); 
    }
}

int initialize(void) 
{
    // function declarations 
    void resize(int, int); 

    // variable declarations 
    PIXELFORMATDESCRIPTOR pfd; 
    int iPixelFormatIndex; 

    // code 
    // 1) pixelformat descriptor initialization 
    ZeroMemory((void*)&pfd, sizeof(PIXELFORMATDESCRIPTOR)); 
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR); 
    pfd.nVersion = 1; 
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER; 
    pfd.iPixelType = PFD_TYPE_RGBA; 
    pfd.cColorBits = 32; // c - count 
    pfd.cRedBits = 8; 
    pfd.cGreenBits = 8; 
    pfd.cBlueBits = 8; 
    pfd.cAlphaBits = 8; 

    // 2) get DC 
    ghdc = GetDC(ghwnd); 
    if(ghdc == NULL) 
    {
        fprintf(gpFile, "getDC() failed\n"); 
        return (-1); 
    }

    // 3) Get matching pixel format index using hdc and pfd 
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd); 
    if(iPixelFormatIndex == 0) 
    {
        fprintf(gpFile, "ChoosePixelFormat() failed\n"); 
        return (-2); 
    }

    // 4) set the pixel format of found index. 
    if(SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
    {
        fprintf(gpFile, "SetPixelFormat() failed\n"); 
        return (-3); 
    } 

    // 5) Create rendering context using hdc, pfd and chosen pixel format index
    ghrc = wglCreateContext(ghdc); // this is not Win32 function. This is WGL (bridging API)  
    if(ghrc == NULL) 
    {
        fprintf(gpFile, "wglCreateContext() failed\n"); 
        return (-4); 
    }
    
    // 6) Make this rendering context as current context 
    if(wglMakeCurrent(ghdc, ghrc) == FALSE) 
    { 
        fprintf(gpFile, "wglMakeCurrent() failed\n"); 
        return (-5); 
    }
    
    // ***** FROM HERE ONWARDS OPENGL CODE STARTS *****  
    // tell OpenGL to choose color to clear the screen 
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 

    // warmup resize 
    resize(WIN_WIDTH, WIN_HEIGHT); 

    return (0); 
}

void resize(int width, int height) 
{
    //  code 
    // if height by accident becomes <=0, make height 1 
    if(height <= 0) 
    {
        height = 1; 
    }

    // Set the view port 
    glViewport(0, 0, (GLsizei)width, (GLsizei)height); 

    // set matrix projection mode 
    glMatrixMode(GL_PROJECTION); /* matrix cha mode projection la thev */

    // set to identity matrix 
    glLoadIdentity(); /* jya matrix cha mode aata projection la set kela tya matrix la atta identity matrix kr */

    // do orthographic projection 
    if(width <= height) 
    {
        glOrtho(        /* asa box tayar kr jyach bottom ani top height ani width shi multiply kel aahe */ 
            -100.0f, 
            100.0f, 
            (-100.0f * ((GLfloat)height / (GLfloat)width)), /* This ratios are taken because when we resize the content should be proportionally resized */
            (100.0f * ((GLfloat)height / (GLfloat)width)), 
            -100.0f, 
            100.0f 
        ); 
        // when wigth < height then top and bottom multiplied by ratio 
	    // when wigth > height then left and right multiplied by ratio 
        // this is done because when the volumn is resized the triangle should also be resized proportionally 
    }
    else 
    {
        glOrtho(
            (-100.0f * ((GLfloat)width / (GLfloat)height)), 
            (100.0f * ((GLfloat)width / (GLfloat)height)), 
            -100.0f, 
            100.0f, 
            -100.0f,
            100.0f 
        ); 
    }
}

void display(void) 
{
    // code 
    // clear OpenGL buffers 
    glClear(GL_COLOR_BUFFER_BIT); 

    // set matrix to modelview to mode 
    glMatrixMode(GL_MODELVIEW); 

    // set it to identity matrix 
    glLoadIdentity(); 
    
    glBegin(GL_TRIANGLES); 
    glVertex3f(0.0f, 50.0f, 0.0f); 
    glVertex3f(-50.0f, -50.0f, 0.0f); 
    glVertex3f(50.0f, -50.0f, 0.0f); 
    glEnd(); 

    // Swap the buffers 
    SwapBuffers(ghdc);  
}

void update(void) 
{
    // code 
}

void uninitialize(void) 
{
    // function declarations 
    void toggleFullScreen(void); 

    // code 

    // If user is exitting in fullscreen then restore fullscreen to normal.
    if(gbFullScreen == TRUE) 
    {
        toggleFullScreen(); 
        gbFullScreen = FALSE; 
    }

    // Make hdc as current context by releasing rendering rendering context as current context 
    if(wglGetCurrentContext() == ghrc) 
    {
        wglMakeCurrent(NULL, NULL); 
    }

    // delete the rendering context 
    if(ghrc) 
    {
        wglDeleteContext(ghrc); 
        ghrc = NULL; 
    }

    // release the DC 
    if(ghdc) 
    {
        ReleaseDC(ghwnd, ghdc); 
        ghdc = NULL; 
    }

    // Destroy Window 
    if(ghwnd) 
    {
        DestroyWindow(ghwnd); 
        ghwnd = NULL; 
    }
    
    // close the file 
    if(gpFile) 
    {
        fprintf(gpFile, "Program terminated successfully\n"); 
        fclose(gpFile); 
        gpFile = NULL; 
    }
}

