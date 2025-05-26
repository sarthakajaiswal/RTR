// standard header files 
#include <Windows.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 

// openGL related header files 
#include <gl\GL.h> 

// standard libraries 
#pragma comment(lib, "user32.lib") 
#pragma comment(lib, "gdi32.lib") 

// openGL related libraries 
#pragma comment(lib, "openGL32.lib")

// macros 
#define WIN_WIDTH   600 
#define WIN_HEIGHT  600 

// circle related macros 
#define DEG2RAD     (3.14 / 180.0)  

// enum related to circle 
enum CircleType 
{
    USING_POINTS = 0, 
    USING_LINES 
}; 

// global variable declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);  
void printCircleType(HWND, HDC); 

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

// circle related Variables 
enum CircleType typeOfCircle = USING_POINTS; 
char strCircleType[16] = "Using GL_POINTS"; 

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

    RECT rc; 

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

                case 'p': 
                case 'P': 
                    typeOfCircle = USING_POINTS; 
                    break; 

                case 'l': 
                case 'L': 
                    typeOfCircle = USING_LINES; 
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
    // function declarations 
    void drawCircleUsingLines(float, float, float); 
    void drawCircleUsingPoints(float, float, float); 
    void printCirleType(void); 

    // variable declarations 
    int numberOfBlueLines = 20; 
    float distanceBetweenTwoLines; 

    float xCoordinateOfVerticalLines = 0.0f;  
    float yCoordinateOfHorizontalLines = 0.0f; 

    // circle related variables 
    float circleRadiusInGraph = 12; /* radius of circle of according to graph scale (1 line = 1)  
                                         should be smaller than numberOfBlueLines */ 
    float circleRadius;             /* radius according to OpenGL scale */

    float cx = 0.0f, cy = 0.0f;     /* center of circle */
                                    /* co-ordinates on circumference at angle theta */

    // code 
    glClear(GL_COLOR_BUFFER_BIT); 

    // --------------- draw graph paper --------------------------  
    // draw graph lines other than central lines 
    distanceBetweenTwoLines = 1.0f / (numberOfBlueLines + 1); 

    glColor3f(0.0f, 0.0f, 1.0f); 
    for(int i = 1; i <= numberOfBlueLines; ++i) 
    {
        xCoordinateOfVerticalLines = xCoordinateOfVerticalLines + distanceBetweenTwoLines; 
        yCoordinateOfHorizontalLines = yCoordinateOfHorizontalLines + distanceBetweenTwoLines; 
        
        if(i%5 == 0)  // for making every fifth line in graph thicker than others 
            glLineWidth(1.5f); 
        else 
            glLineWidth(1.0f); 

        glBegin(GL_LINES); 
        // horizontal lines 
        glVertex3f(1.0f, yCoordinateOfHorizontalLines, 0.0f); 
        glVertex3f(-1.0f, yCoordinateOfHorizontalLines, 0.0f); 

        glVertex3f(1.0f, -yCoordinateOfHorizontalLines, 0.0f); 
        glVertex3f(-1.0f, -yCoordinateOfHorizontalLines, 0.0f); 

        // vertical lines 
        glVertex3f(xCoordinateOfVerticalLines, 1.0f, 0.0f); 
        glVertex3f(xCoordinateOfVerticalLines, -1.0f, 0.0f); 

        glVertex3f(-xCoordinateOfVerticalLines, 1.0f, 0.0f); 
        glVertex3f(-xCoordinateOfVerticalLines, -1.0f, 0.0f); 

        glEnd();  
    }

    // draw central lines of graph 
    glLineWidth(2.0f); 
    
    glBegin(GL_LINES); 
    // y-axis 
    glColor3f(0.0f, 1.0f, 0.0f); 
    glVertex3f(0.0f, -1.0f, 0.0f); 
    glVertex3f(0.0f, 1.0f, 0.0f); 
    
    // x-axis 
    glColor3f(1.0f, 0.0f, 0.0f); 
    glVertex3f(-1.0f, 0.0f, 0.0f); 
    glVertex3f(1.0f, 0.0f, 0.0f); 
    glEnd();

    glLineWidth(1.0f); 

    // print circle type 
    printCircleType(ghwnd, ghdc); 

    // ---------------- draw circle ------------------   
    circleRadius = circleRadiusInGraph * distanceBetweenTwoLines; 

    if(typeOfCircle == USING_POINTS) 
    {
        drawCircleUsingPoints(cx, cy, circleRadius); 
        sprintf(strCircleType, "GL_POINTS");  
    }
    else 
    {
        drawCircleUsingLines(cx, cy, circleRadius); 
        sprintf(strCircleType, "GL_LINES"); 
    } 


    SwapBuffers(ghdc); 
}

void printCircleType(HWND hwnd, HDC hdc) 
{
    // variables 
    RECT rc; 

    // code 
    GetClientRect(hwnd, &rc); 
    SetBkColor(hdc, RGB(0, 0, 0)); 
    SetTextColor(hdc, RGB(0, 255, 0)); 
    DrawText(hdc, strCircleType, -1, &rc, DT_SINGLELINE | DT_CENTER | DT_BOTTOM);
}

void drawCircleUsingLines(float cx, float cy, float radius) 
{
    // variable declarations 
    float xOnCircumference, yOnCircumference; 
    float theta; 

    // code 
    glColor3f(1.0f, 1.0f, 0.0f); 
    glBegin(GL_LINE_LOOP); 
    for(float angle = 0.0f; angle < 360; angle = angle + 1) 
    {
        theta = angle * DEG2RAD; 
        xOnCircumference = cx + (radius * cos(theta)); 
        yOnCircumference = cy + (radius * sin(theta)); 

        glVertex3f(xOnCircumference, yOnCircumference, 0.0f); 
    }
    glEnd(); 
}

void drawCircleUsingPoints(float cx, float cy, float radius) 
{
    // variable declarations 
    float xOnCircumference, yOnCircumference; 
    float theta; 
    
    // code 
    glColor3f(1.0f, 1.0f, 0.0f); 
    glBegin(GL_POINTS); 
    for(float angle = 0.0f; angle < 360; angle = angle + 0.05) 
    {
        theta = angle * DEG2RAD; 
        xOnCircumference = cx + (radius * sin(theta)); 
        yOnCircumference = cy + (radius * cos(theta)); 

        glVertex3f(xOnCircumference, yOnCircumference, 0.0f); 
    }
    glEnd(); 
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

