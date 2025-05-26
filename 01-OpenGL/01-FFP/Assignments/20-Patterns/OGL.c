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

struct Color {
    float r, g, b, a; 
}; 

// global function declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);  

void pattern1(void); 
void pattern2(void); 
void pattern3(void); 
void pattern4(void); 
void pattern5(void); 
void pattern6(void); 

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

// pattern related variables 
float basePointsCoordinates[4][4][2]; 

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
    wnd.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
    wnd.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON)); 
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

    // fill the base points coordinates matrix 
    float startX = -0.6; 
    float startY = 0.6; 

    float x = startX, y = startY; 
    for(int i = 0; i < 4; ++i) 
    {
        x = startX; 
        for(int j = 0; j < 4; ++j) 
        {
            basePointsCoordinates[i][j][0] = x; 
            basePointsCoordinates[i][j][1] = y; 

            x = x + 0.4; 
        }
        y = y - 0.4; 
    }

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
    // code 
    glClear(GL_COLOR_BUFFER_BIT); 

    glLineWidth(2.0); 
    // pattern 1 : 16 dots 
    glPushMatrix(); 
    {
        glTranslatef(-0.6f, 0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern1(); 
    }
    glPopMatrix(); 

    // pattern 2 
    glPushMatrix(); 
    {   
        glTranslatef(0.0f, 0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern2();
    }
    glPopMatrix(); 

    // pattern 3
    glPushMatrix(); 
    {   
        glTranslatef(0.6f, 0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern3();
    }
    glPopMatrix(); 

    // pattern 4 
    glPushMatrix(); 
    {   
        glTranslatef(-0.6f, -0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern4(); 
    }
    glPopMatrix(); 

    // pattern 5 
    glPushMatrix(); 
    {   
        glTranslatef(0.0f, -0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern5();
    }
    glPopMatrix(); 

    // pattern 6 
    glPushMatrix(); 
    {   
        glTranslatef(0.6f, -0.5f, 0.0f); 
        glScalef(0.4, 0.4, 0.0);  
        pattern6();
    }
    glPopMatrix(); 

    glLineWidth(1.0); 

    SwapBuffers(ghdc); 
}

void update(void) 
{
    //code 
}

void pattern1(void) 
{
    // code 
    glEnable(GL_POINT_SMOOTH); 
    glPointSize(4.0); 

    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_POINTS); 
    for(int i = 0; i < 4; ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f); 
        }
    }
    glEnd(); 
    glPointSize(1.0); 
    glDisable(GL_POINT_SMOOTH);
} 

void pattern2(void) 
{
    // code 
    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_LINES); 
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j) 
        {
            // line to previous point in same row
            if(j > 0 && i < 3)
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i][j-1][0], basePointsCoordinates[i][j-1][1], 0.0f); 
            }
            
            // line to the diagonally backward (previous position + in bottom row)
            if(j > 0 && i < 3) 
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i+1][j-1][0], basePointsCoordinates[i+1][j-1][1], 0.0f);  
            } 

            // line to below point in same column 
            if(j < 3 && i < 3)
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i+1][j][0], basePointsCoordinates[i+1][j][1], 0.0f);  
            }
        }
    }
    glEnd(); 
}

void pattern3(void) 
{
    // code 
    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_LINES); 
    for(int i = 0, j = 0; i < 4, j < 4; ++i, ++j)
    {    
        glVertex3f(basePointsCoordinates[i][0][0], basePointsCoordinates[i][0][1], 0.0f); 
        glVertex3f(basePointsCoordinates[i][3][0], basePointsCoordinates[i][3][1], 0.0f); 

        glVertex3f(basePointsCoordinates[0][j][0], basePointsCoordinates[0][0][1], 0.0f); 
        glVertex3f(basePointsCoordinates[3][j][0], basePointsCoordinates[3][j][1], 0.0f); 
    }
    glEnd(); 
}

void pattern4(void) 
{
    // code 
    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_LINES); 
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j) 
        {
            // line to previous point in same row
            if(j > 0)
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i][j-1][0], basePointsCoordinates[i][j-1][1], 0.0f); 
            }
            
            // line to the diagonally backward (previous position + in bottom row)
            if(j > 0 && i < 3) 
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i+1][j-1][0], basePointsCoordinates[i+1][j-1][1], 0.0f);  
            } 

            // line to below point in same column 
            if(i < 3)
            {
                glVertex3f(basePointsCoordinates[i][j][0], basePointsCoordinates[i][j][1], 0.0f);
                glVertex3f(basePointsCoordinates[i+1][j][0], basePointsCoordinates[i+1][j][1], 0.0f);  
            }
        }
    }
    glEnd();
}

void pattern5(void) 
{
    // code 
    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_LINES); 

    // outer square right and bottom border 
    glVertex3f(basePointsCoordinates[0][3][0], basePointsCoordinates[0][3][1], 0.0f);
    glVertex3f(basePointsCoordinates[3][3][0], basePointsCoordinates[3][3][1], 0.0f);
    
    glVertex3f(basePointsCoordinates[3][3][0], basePointsCoordinates[3][3][1], 0.0f);
    glVertex3f(basePointsCoordinates[3][0][0], basePointsCoordinates[3][0][1], 0.0f);

    // lines originating from vertex[0][0]
    for(int i = 0; i < 4; ++i)
    {
        glVertex3f(basePointsCoordinates[0][0][0], basePointsCoordinates[0][0][1], 0.0f);
        glVertex3f(basePointsCoordinates[i][3][0], basePointsCoordinates[i][3][1], 0.0f);
        
        glVertex3f(basePointsCoordinates[0][0][0], basePointsCoordinates[0][0][1], 0.0f);
        glVertex3f(basePointsCoordinates[3][i][0], basePointsCoordinates[3][i][1], 0.0f);
    }

    glEnd(); 
}

void pattern6(void) 
{
    // variable declarations 
    struct Color pattern6Colors[3] = {
        {1.0, 0.0, 0.0, 1.0}, 
        {0.0, 1.0, 0.0, 1.0}, 
        {0.0, 0.0, 1.0, 1.0}
    }; 

    // code 
    for(int i = 0; i < 3; ++i) 
    {
        glColor3f(pattern6Colors[i].r, pattern6Colors[i].g, pattern6Colors[i].b); 
        glBegin(GL_QUADS); 
        glVertex3f(basePointsCoordinates[0][i][0], basePointsCoordinates[0][i][1], 0.0f); 
        glVertex3f(basePointsCoordinates[3][i][0], basePointsCoordinates[3][i][1], 0.0f); 
        glVertex3f(basePointsCoordinates[3][i+1][0], basePointsCoordinates[3][i+1][1], 0.0f); 
        glVertex3f(basePointsCoordinates[0][i+1][0], basePointsCoordinates[0][i+1][1], 0.0f); 
        glEnd(); 
    }
    /* pattern 4*/
    pattern3(); 
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

