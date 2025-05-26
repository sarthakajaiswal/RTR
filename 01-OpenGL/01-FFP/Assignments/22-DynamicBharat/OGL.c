// standard header files 
#include <Windows.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <playsoundapi.h> 

#include <math.h> 

// openGL related header files 
#include <gl\GL.h>  

// custom header files 
#include "OGL.h" 

// standard libraries 
#pragma comment(lib, "user32.lib") 
#pragma comment(lib, "gdi32.lib") 

// openGL related libraries 
#pragma comment(lib, "openGL32.lib")

// audio relared libraries 
#pragma comment(lib, "Winmm.lib") 

// macros 
#define WIN_WIDTH   800 
#define WIN_HEIGHT  600 

#define DEG2RAD (3.14/180.0)

struct Color 
{
    float r, g, b, a; 
};  

struct Position{
    float x, y, z; 
}; 

// global function declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);  
void playAudio(void); 

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
BOOL gbEscapeKeyIsPressed = FALSE; 

// OpenGL related global variables 
HDC ghdc = NULL; 
HGLRC ghrc = NULL; 

// gameloop related variables 
BOOL bDone = FALSE; 

float tx = 0.0f;

//object related variables 
float aeroplaneX = -1.2f; 

float B_Alpha = 0.0; 
float H_Alpha = 0.0; 
float A1_Alpha = 0.0; 
float A2_Alpha = 0.0; 
float R_Alpha = 0.0; 
float T_Alpha = 0.0; 

struct Position posOfB = {-1.4, 0.0}; 
struct Position posOfH = {-0.8, 1.4}; 
struct Position posOfA1 = {-0.8, -1.4}; 
struct Position posOfR = {0.8, 1.4}; 
struct Position posOfA2 = {0.8, -1.4}; 
struct Position posOfT = {1.2, 0.0}; 

struct Color normalColorOfText = {0.7f, 0.7f, 0.7f}; 

// variables controlling speed 
float textTranslationSpeed = 0.0005; // text speed  
static float deltaAeroplaneX = 0.0003;        // aeroplane speed 

// side plane related variables 

float upperPlaneBazierBeginX = -1.0, upperPlaneBazierBeginY = 1.0; 
float upperPlaneBazierControlX = -1.0f, upperPlaneBazierControlY = 0.15f; 
float upperPlaneBazierEndX = -0.67f, upperPlaneBazierEndY = 0.05f; 
float upperPlaneBazierT = 0.0f; 
float x, y; 
float upperPlaneRotateAngle = 90.0f; 

float upperPlaneX = -1.2f; 
float upperPlaneY = 1.2; 

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

                case '+': 
                    tx += 0.01; 
                    break; 
                
                case '-': 
                    tx -= 0.01; 
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

    toggleFullScreen(); 
    playAudio(); 

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

void playAudio(void) 
{
    char buffer[MAX_PATH] = {0}; 
    GetModuleFileName(NULL, buffer, MAX_PATH); 
    char *LastSlash = strrchr(buffer, '\\'); 
    if(LastSlash == NULL) 
        LastSlash = strrchr(buffer, '/'); 
    buffer[LastSlash - buffer] = 0; 
    strcat(buffer, "\\YeDeshHaiVeerJawanoKa.wav"); 
    PlaySound(buffer, NULL, SND_ASYNC | SND_LOOP); 
}

void display(void) 
{
    // local function declarations 
    void b(void); 
    void h(void); 
    void a(float); 
    void r(void); 
    void t(void); 

    void aeroplane(struct Color);  

    // code 
    char title[128]; 
    glClear(GL_COLOR_BUFFER_BIT); 
    
    glEnable(GL_BLEND); 
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    
    glPushMatrix(); 
    {
        glTranslatef(posOfB.x, posOfB.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        b(); 
    }
    glPopMatrix(); 
    glPushMatrix(); 
    {
        glTranslatef(posOfH.x, posOfH.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        h(); 
    }
    glPopMatrix(); 

    glPushMatrix(); 
    {
        glTranslatef(posOfA1.x, posOfA1.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        a(A1_Alpha); 
    }
    glPopMatrix(); 

    glPushMatrix(); 
    {
        glTranslatef(posOfR.x, posOfR.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        r(); 
    }
    glPopMatrix(); 

    glPushMatrix(); 
    {
        glTranslatef(posOfA2.x, posOfA2.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        a(A2_Alpha); 
    }
    glPopMatrix(); 

    glPushMatrix(); 
    {
        glTranslatef(posOfT.x, posOfT.y, 0.0); 
        glScalef(0.25, 0.5, 1.0); 
        t(); 
    }
    glPopMatrix(); 
    
    struct Color orange = {1.0, 0.5, 0.25}; 
    struct Color white = {1.0, 1.0, 1.0}; 
    struct Color green = {0.0f, 1.0f, 0.0f}; 
    
    glPushMatrix(); 
    {  
        glTranslatef(upperPlaneX-0.05, upperPlaneY+0.06, 0.0); 
        glScalef(0.15, 0.2, 0.2); 
        
        glRotatef(-90 - upperPlaneRotateAngle, 0, 0, 1); 
        
        aeroplane(orange); 
    } 
    glPopMatrix(); 
    
    glPushMatrix(); 
    {   
        glTranslatef(upperPlaneX-0.05, -upperPlaneY-0.1, 0.0); 
        glScalef(0.15, 0.2, 0.2); 
       
        glRotatef((-90 + upperPlaneRotateAngle), 0, 0, 1); 
       
        aeroplane(green); 
    } 
    glPopMatrix(); 

    glPushMatrix(); 
    {
        glTranslatef(aeroplaneX, -0.02, 0.0); 
        glScalef(0.15, 0.2, 0.2); 
        glRotatef(-90, 0, 0, 1); 
        aeroplane(white); 
    } 
    glPopMatrix(); 
    
    glDisable(GL_BLEND); 
    SwapBuffers(ghdc); 
}

void changeBazierVariablesForUpperPlaneGoingUpward(void) 
{
    static int isCalledAlready = 0; 

    if(isCalledAlready == 1) // to ensure function is called only once 
        return; 

    isCalledAlready += 1; 
    upperPlaneBazierBeginX = 0.62, upperPlaneBazierBeginY = 0.05; 
    upperPlaneBazierControlX = 1.0f, upperPlaneBazierControlY = 0.15f; 
    upperPlaneBazierEndX = 1.0f, upperPlaneBazierEndY = 1.0f; 
    upperPlaneBazierT = 0.0f; 
}

void update(void) 
{
    // variable declarations 
    static float deltaAlpha = 0.001;  

    // code
    // alphabet translation 
    if(posOfB.x <= -0.67) 
    {
        posOfB.x += (-0.67-(-1.4)) * textTranslationSpeed; 
    }
    else if(posOfH.x <= -0.37f) 
    {
        posOfH.x += (-0.37-(-0.8)) * textTranslationSpeed; 
        posOfH.y -= (1.4) * textTranslationSpeed; 
    }
    else if(posOfA1.x <= -0.12f)
    {
        posOfA1.x += (-0.12-(-0.8)) * textTranslationSpeed; 
        posOfA1.y += (1.4) * textTranslationSpeed; 
    }
    else if(posOfR.x >= 0.12) 
    {
        posOfR.x -= (0.8-0.12) * textTranslationSpeed; 
        posOfR.y -= (1.4) * textTranslationSpeed; 
    }
    else if(posOfA2.x >= 0.37) 
    {
        posOfA2.x -= (0.8-0.37) * textTranslationSpeed; 
        posOfA2.y += (1.4) * textTranslationSpeed; 
    }
    else if(posOfT.x >= 0.62)
    {
        posOfT.x -= (1.4-0.62) * textTranslationSpeed; 
    }

    else 
    {
        aeroplaneX += deltaAeroplaneX; 

        if(aeroplaneX <= -0.67) 
        {
            upperPlaneX = ((1-upperPlaneBazierT)*((1-upperPlaneBazierT)*upperPlaneBazierBeginX + upperPlaneBazierT*upperPlaneBazierControlX) + (upperPlaneBazierT * ((1-upperPlaneBazierT) * upperPlaneBazierControlX + upperPlaneBazierT * upperPlaneBazierEndX))); 
            upperPlaneY = ((1-upperPlaneBazierT)*((1-upperPlaneBazierT)*upperPlaneBazierBeginY + upperPlaneBazierT*upperPlaneBazierControlY) + (upperPlaneBazierT * ((1-upperPlaneBazierT) * upperPlaneBazierControlY + upperPlaneBazierT * upperPlaneBazierEndY)));

            upperPlaneBazierT += 1.0 / ((-0.67-(-1.2))/deltaAeroplaneX); // formula to match middle and side aeroplanes after turn of side aeroplane 
            upperPlaneRotateAngle -= 90.0 / ((-0.67-(-1.2))/deltaAeroplaneX); 
        }
        else if(aeroplaneX <= 0.62)
        {
            upperPlaneX = aeroplaneX; 
            upperPlaneY = 0.05; 
        }
        else 
        {
            changeBazierVariablesForUpperPlaneGoingUpward(); 
            upperPlaneX = ((1-upperPlaneBazierT)*((1-upperPlaneBazierT)*upperPlaneBazierBeginX + upperPlaneBazierT*upperPlaneBazierControlX) + (upperPlaneBazierT * ((1-upperPlaneBazierT) * upperPlaneBazierControlX + upperPlaneBazierT * upperPlaneBazierEndX))); 
            upperPlaneY = ((1-upperPlaneBazierT)*((1-upperPlaneBazierT)*upperPlaneBazierBeginY + upperPlaneBazierT*upperPlaneBazierControlY) + (upperPlaneBazierT * ((1-upperPlaneBazierT) * upperPlaneBazierControlY + upperPlaneBazierT * upperPlaneBazierEndY)));
            upperPlaneBazierT += 1.0 / ((-0.67-(-1.2))/deltaAeroplaneX);
            
            upperPlaneRotateAngle -= 90.0 / ((-0.67-(-1.2))/deltaAeroplaneX); 
        }

        // color change 
        if(aeroplaneX >= -0.57 && B_Alpha < 1.0)  
        {
            B_Alpha += deltaAlpha; 
        }
        if(aeroplaneX >= -0.32 && H_Alpha < 1.0)  
        {
            H_Alpha += deltaAlpha; 
        }
        if(aeroplaneX >= -0.07 && A1_Alpha < 1.0)  
        {
            A1_Alpha += deltaAlpha; 
        }
        if(aeroplaneX >= 0.18 && R_Alpha < 1.0)  
        {
            R_Alpha += deltaAlpha; 
        }
        if(aeroplaneX >= 0.42 && A2_Alpha < 1.0)  
        {
            A2_Alpha += deltaAlpha; 
        }
        if(aeroplaneX >= 0.67 && T_Alpha < 1.0)  
        {
            T_Alpha += deltaAlpha;  
        }
    }

    if(aeroplaneX >= 1.2) 
        bDone = TRUE; 
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

void b(void) 
{
    // variable declarations 
    float Quads[][4][2] = {
                            {{-0.3, 0.0}, {-0.3, -0.05}, {0.2, -0.05}, {0.3, 0.0}},  
                            {{-0.3, -0.05}, {-0.3, -0.1}, {0.3, -0.1}, {0.2, -0.05}},  
                            
                            {{0.2, 0.0}, {0.3, 0.0}, {0.3, 0.3}, {0.2, 0.3}},  
                            {{-0.3, -0.0}, {-0.2, 0.0}, {-0.2, 0.3}, {-0.3, 0.3}},  
                            {{-0.4, 0.3}, {0.3, 0.3}, {0.2, 0.4}, {-0.4, 0.4}},  
                            
                            {{-0.3, -0.4}, {-0.2, -0.4}, {-0.2, -0.1}, {-0.3, -0.1}},  
                            {{0.2, -0.4}, {0.3, -0.4}, {0.3, -0.1}, {0.2, -0.1}},  
                            {{-0.4, -0.5}, {0.2, -0.5}, {0.3, -0.4}, {-0.4, -0.4}},  
                        }; 

    float ColorsOfQuads[][4][3] = {
                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 

                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 

                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
                        }; 

    glBegin(GL_QUADS); 

    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(normalColorOfText.r, normalColorOfText.g, normalColorOfText.b, 1.0f); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        }
    }

    for(int i = 0; i < 11; ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(ColorsOfQuads[i][j][0], ColorsOfQuads[i][j][1], ColorsOfQuads[i][j][2], B_Alpha); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        }
    }

    glEnd(); 
}

void h(void) 
{
    // variable declarations 
    float Quads[][4][2] = {
                            {{0.25, 0.3}, {0.25, 0.4}, {0.05, 0.4}, {0.05, 0.3}},
                            {{-0.15, 0.3}, {-0.15, 0.4}, {-0.35, 0.4}, {-0.35, 0.3}},
                            {{-0.2, 0.3}, {-0.3, 0.3}, {-0.3, 0.0}, {-0.2, 0.0}}, 
                            {{0.2, 0.3}, {0.1, 0.3}, {0.1, 0.0}, {0.2, 0.0}},
                            
                            {{-0.3, 0.0}, {-0.3, -0.1}, {0.2, -0.1}, {0.2, 0.0}},   
                            
                            {{0.1, -0.4}, {0.2, -0.4}, {0.2, -0.1}, {0.1, -0.1}}, 
                            {{-0.3, -0.4}, {-0.2, -0.4}, {-0.2, -0.1}, {-0.3, -0.1}}, 
                            {{0.25, -0.5}, {0.25, -0.4}, {0.05, -0.4}, {0.05, -0.5}},
                            {{-0.15, -0.5}, {-0.15, -0.4}, {-0.35, -0.4}, {-0.35, -0.5}},   
                        }; 

    float ColorsOfQuads[][4][3] = {
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            
                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
                        }; 

    glBegin(GL_QUADS); 
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(normalColorOfText.r, normalColorOfText.g, normalColorOfText.b, 1.0f); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        } 
    }
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(ColorsOfQuads[i][j][0], ColorsOfQuads[i][j][1], ColorsOfQuads[i][j][2], H_Alpha); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        } 
    }
    glEnd(); 
}

void a(float ColorAlpha) 
{
    // variable declarations 
    float Quads[][4][2] = {
                            {{-0.2, 0.3}, {0.1, 0.3}, {0.0, 0.4}, {-0.1, 0.4}}, 
                            {{-0.05, 0.3}, {-0.2, 0.3}, {-0.3, 0.0}, {-0.2, 0.0}}, 
                            {{0.1, 0.3}, {-0.05, 0.3}, {0.1, 0.0}, {0.2, 0.0}}, 
                            
                            {{-0.3, -0.1}, {0.2, -0.1}, {0.2, 0.0}, {-0.3, 0.0}}, 

                            {{0.1, -0.4}, {0.2, -0.4}, {0.2, -0.1}, {0.1, -0.1}}, 
                            {{-0.3, -0.4}, {-0.2, -0.4}, {-0.2, -0.1}, {-0.3, -0.1}}, 
                            {{-0.15, -0.4}, {-0.35, -0.4}, {-0.35, -0.5}, {-0.15, -0.5}}, 
                            {{0.05, -0.4}, {0.05, -0.5}, {0.25, -0.5}, {0.25, -0.4}} 
                        }; 
    float ColorsOfQuads[][4][3] = {
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 

                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}} 
                        }; 

    glBegin(GL_QUADS); 
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(normalColorOfText.r, normalColorOfText.g, normalColorOfText.b, 1.0f); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        } 
    }                    
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(ColorsOfQuads[i][j][0], ColorsOfQuads[i][j][1], ColorsOfQuads[i][j][2], ColorAlpha); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        }
    }

    glEnd(); 
}

void r(void) 
{
    // variable declarations 
    float Quads[][4][2] = { 
                            {{-0.4, 0.3}, {0.3, 0.3}, {0.2, 0.4}, {-0.4, 0.4}}, 
                            {{-0.2, 0.3}, {-0.3, 0.3}, {-0.3, 0.0}, {-0.2, 0.0}}, 
                            {{0.3, 0.3}, {0.1, 0.3}, {0.2, 0.0}, {0.3, 0.0}}, 
                            
                            {{-0.3, 0.0}, {-0.3, -0.1}, {0.2, -0.1}, {0.3, 0.0}}, 
                            
                            {{-0.3, -0.4}, {-0.2, -0.4}, {-0.2, -0.1}, {-0.3, -0.1}}, 
                            {{0.1, -0.4}, {0.2, -0.4}, {0.0, -0.1}, {-0.1, -0.1}}, 
                            {{0.05, -0.4}, {0.05, -0.5}, {0.25, -0.5}, {0.25, -0.4}}, 
                            {{-0.35, -0.4}, {-0.35, -0.5}, {-0.15, -0.5}, {-0.15, -0.4}}, 
                        }; 
    float ColorsOfQuads[][4][3] = {
                        {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                        {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                        {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 

                        {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 

                        {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                        {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                        {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
                        {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 
    }; 

    glBegin(GL_QUADS); 
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(normalColorOfText.r, normalColorOfText.g, normalColorOfText.b, 1.0f); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        } 
    }
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(ColorsOfQuads[i][j][0], ColorsOfQuads[i][j][1], ColorsOfQuads[i][j][2], R_Alpha); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        }
    }
    glEnd(); 
}

void t(void) 
{
    // variable declarations 
    float Quads[][4][2] = {
                            {{-0.4, 0.4}, {-0.4, 0.3}, {0.3, 0.3}, {0.3, 0.4}},  
                            {{0.0, 0.3}, {-0.1, 0.3}, {-0.1, 0.0}, {0.0, 0.0}}, 
                            
                            {{-0.1, 0.0}, {-0.1, -0.1}, {0.0, -0.1}, {0.0, 0.0}}, 
                            
                            {{-0.1, -0.4}, {0.0, -0.4}, {0.0, -0.1}, {-0.1, -0.1}}, 
                            {{-0.15, -0.4}, {-0.15, -0.5}, {0.05, -0.5}, {0.05, -0.4}}, 
                        }; 
    float ColorsOfQuads[][4][3] = {
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}}, 
                            {{1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            
                            {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 

                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, 
                            {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}  
                        }; 

    glBegin(GL_QUADS); 
    for(int i = 0; i < (sizeof(Quads)/sizeof(Quads[0])); ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(normalColorOfText.r, normalColorOfText.g, normalColorOfText.b, 1.0f); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        } 
    }
    for(int i = 0; i < 5; ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            glColor4f(ColorsOfQuads[i][j][0], ColorsOfQuads[i][j][1], ColorsOfQuads[i][j][2], T_Alpha); 
            glVertex2f(Quads[i][j][0], Quads[i][j][1]); 
        }
    }
    glEnd(); 
}

void aeroplane(struct Color tailColor) 
{
    // function declarations 
    void aeroplaneRightSide(struct Color); 

    aeroplaneRightSide(tailColor); 

    glPushMatrix(); 
    {
        glScalef(-1.0f, 1.0, 0.0); 
        aeroplaneRightSide(tailColor); 
    }
    glPopMatrix(); 
}

void aeroplaneRightSide(struct Color tailColor) 
{
    // function declarations 
    void drawFilledCircle(float, float, float); 

    // variable declarations 
    float QuadsOfAeroplane[][4][2] = {
        {{0.1, 0.5}, {0.0 ,0.5}, {0.0, -0.67}, {0.05, -0.67}}, 
        
        {{0.05, 0.4}, {0.7, -0.2}, {0.7, -0.4}, {0.05, -0.2}},  
        {{0.05, -0.25}, {0.05, -0.6}, {0.15, -0.7}, {0.35, -0.7}}, 
        
        {{0.5, 0.0}, {0.5, -0.2}, {0.54, -0.2}, {0.54, -0.0}}, 
        {{0.3, 0.2}, {0.3, -0.1}, {0.34, -0.1}, {0.34, 0.2}} 
    }; 

    float TrianglesOfAeroplane[][3][2] = {
            {{0.1, 0.5}, {0.0, 0.5}, {0.0, 1.0f}}, 
            {{0.7, -0.2}, {0.7, -0.4}, {0.8, -0.4}}, 
            {{0.15, -0.7}, {0.35, -0.7}, {0.2, -0.8}}
        }; 

    float LinesInAeroplane[][2][2] = {
        {{0.1, 0.4}, {0.07, -0.2}}, 
    }; 

    // code 
    for(int i = 0; i < 6; ++i) 
    {
        glColor3f(64.0/255, 127.0/255, 127.0/255);
        glBegin(GL_QUADS); 
        for(int j = 0; j < 4; ++j) 
        {
            if(i == 0 && (j == 1 || j == 2))
            glColor3f(84.0/255, 147.0/255, 147.0/255);
        
            glVertex2f(QuadsOfAeroplane[i][j][0], QuadsOfAeroplane[i][j][1]); 
        }
        glEnd(); 
    }
    
    glBegin(GL_TRIANGLES); 
    for(int i = 0; i < 3; ++i) 
    {
        glColor3f(64.0/255, 127.0/255, 127.0/255);
        for(int j = 0; j < 3; ++j) 
        {
            if(i == 0 && (j == 1 || j == 2))
                glColor3f(84.0/255, 147.0/255, 147.0/255);

            glVertex2f(TrianglesOfAeroplane[i][j][0], TrianglesOfAeroplane[i][j][1]); 
        }
    }
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f); 
    glBegin(GL_LINES); 
    for(int i = 0; i < 1; ++i) 
    {
        for(int j = 0; j < 2; ++j) 
        {
            glVertex2f(LinesInAeroplane[i][j][0], LinesInAeroplane[i][j][1]); 
        }
    }
    glEnd();

    // color tail 
    glColor3f(tailColor.r, tailColor.g, tailColor.b); 
    glBegin(GL_QUADS); 
    glVertex2f(0.05, -0.67); 
    glVertex2f(0.0, -0.67); 
    glVertex2f(0.0, -0.7); 
    glVertex2f(0.02, -0.7); 
    glEnd(); 

    // smoke from aeroplane 
    glBegin(GL_QUADS); 
    glColor4f(tailColor.r, tailColor.g, tailColor.b, 0.5f); 
    glVertex2f(0.02f, -0.7f); 
    glVertex2f(0.0f, -0.7f); 
    glColor4f(0.0f, 0.0f, 0.0f, 0.25f); 
    glVertex2f(0.0f, -0.8f); 
    glVertex2f(0.04f, -0.8f); 
    glEnd(); 
    
    // top nose line 
    glBegin(GL_LINES); 
    glColor3f(1.0f, 1.0f, 1.0f); 
    glVertex2f(0.0f, 1.0f); 
    glVertex2f(0.0f, 1.1f); 
    glEnd(); 

    
    glColor3f(1.0f, 0.5f, 0.25f); 
    drawFilledCircle(0.6, -0.25, 0.045); 
    glColor3f(1.0f, 1.0f, 1.0f); 
    drawFilledCircle(0.6, -0.25, 0.03); 
    glColor3f(0.0f, 1.0f, 0.0f); 
    drawFilledCircle(0.6, -0.25, 0.015); 
}

void drawFilledCircle(float cx, float cy, float r) 
{
    // variable declarations 
    float angle, x, y; 

    // code 
    glBegin(GL_LINES); 
    for(angle = 0; angle <= 360; angle += 1) 
    {
        x = cx + r * cos(angle * DEG2RAD); 
        y = cy + r * sin(angle * DEG2RAD); 

        glVertex2f(x, y); 
  