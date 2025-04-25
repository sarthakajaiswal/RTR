// standard header files 
#include <Windows.h> 
#include <stdio.h>      // file fopen(), fwrite() and fclose() 
#include <stdlib.h>     // exit() 

// openGL related files
#include <gl/GL.h>      // C:\Program Files x86\Windows Kits <- this is our SDK 
                        // C:\Program Files x86\Windows Kits\10\include\version\um\

// custom header files 
#include "OGL.h" 

// OpenGL related libraries 
#pragma comment(lib, "opengl32.lib")  // .dll la dynamically kuthe ani kashi chitkwach yachi mahiti .lib file madhe aste 
                                        // mhanun lib file la import library suddha mhantat 
                                        // C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64

                                        // system32->64 bit chya dll thevte !! 
                                        // system16->32 bit chya dll thevte !! 

                                        // OS swata waprnari dll 
                                        // C:\Windows\system32 madhe thevte  

                                        // Two ways to give lib file - 
                                        // 1) using command line 2) #pragma 

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
        exit(0);    // system kadoon aaleli error aahe, aapli nahi mhanun -1 nahi, 0 exit value  
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

    ghwnd = hwnd;   // saving hwnd into global variable so that it can be used in toggleFullScreen() 
                    /* This is how two function can communicate with each other using global variable 
                         even if one do not calls other (so no way to pass parameter). */ 

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

        // here file is not closed because 
        /*
            DestroyWindow-> WM_DESTROY ->message pool-> game loop -> WM_DESTROY handeler -> post WM_QUIT 
            -> game loop : bDone = FALSE -> loop end in next iteration -> call to uninitialize() -> file close. 
        */
    }
    else 
    {
        fprintf(gpFile, "initialize() completed successfully\n"); 
    }

    // set this window as foreground window and active window 
    /*
        WS_OVERLAPPEDWINDOW itself does this task 
        But if this window is made back by OS in then this two functions regains the focus and foregroundness of window 
    */
    SetForegroundWindow(hwnd); 
    SetFocus(hwnd); 

    // game loop 
    while(bDone == FALSE) 
    {
        if(PeekMessage(&msg, NULL, 0, 0,PM_REMOVE)) // pm_remove -> remove taken message from queue 
        {
            if(msg.message == WM_QUIT) // wm_quit -> press close button | select close in system menu | alt + f4
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
            ZeroMemory((void*)&wpPrev, sizeof(WINDOWPLACEMENT)); 
            wpPrev.length = sizeof(WINDOWPLACEMENT); 
            break; 
        
        case WM_SETFOCUS: 
            gbActiveWindow = TRUE; 
            break; 

        case WM_KILLFOCUS: 
            gbActiveWindow = FALSE; 
            break; 

        case WM_ERASEBKGND: // this case is not  compulsory/necessory but for flicker free rendering this is done. 
                        /*
                            In retained mode rendering WM_PAINT was doing erasing of background internally as part of its implementation. 
                            But now we are using external renderer - openGL, vulkan, metal, so erase background will be done by this renderers 
                            So to say to OS that do not pass this message to DefWindowProc which will further pass it to WM_PAINT we say return(0) 
                        */
            return(0); 
                // return in case is  used only when any case is not expected to go through DefWindowProc() 

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

/* we are using mixed style here Hungarian + Camel */
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
    // variable declarations 
    PIXELFORMATDESCRIPTOR pfd; /* color gun dware pixel la ks rangwach yachi mahiti OS kade aste 
                                    os -> driver-> graphics card -> color gun 
                                
                                    Hi mahiti 
                                    Windows -> PixelFormatDescriptor 
                                    Linux -> FrameBufferAttributes 
                                    Mac -> PixelFormatAtt

                                    Here we are saying openGL's wish about how to glow pixels 
                                */
    int iPixelFormatIndex; 

    // code 
    // 1) pixelformat descriptor initialization 
    ZeroMemory((void*)&pfd, sizeof(PIXELFORMATDESCRIPTOR)); 
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR); 
    pfd.nVersion = 1;   /* Microsoft restricts it to 1 for bussiness  
                            i.e. to suppress others technology (openGL) and promote 
                            its own technology DirectX. 
                            
                        AMD NVIDEA gives full support to openGL, but in ProgrammablePipe
                    */
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER; 
                    /*
                        PFD_DRAW_TO_WINDOW -> window vr drawing krt aahe (printer vr nahi) 
                        PFD_DOUBLEBEFFER -> allow double buffer concept 
                            samor asnara buffer -> front buffer 
                            maghe asnara buffer -> back buffer 
                    */
    pfd.iPixelType = PFD_TYPE_RGBA; 
                    /*
                        openGL says my pixels will have R, G, B and A too. 
                    */
    pfd.cColorBits = 32; // c - count // pixel la ekoon kiti bit deu  
    pfd.cRedBits = 8;   // tyatoon r la kiti byte deu 
    pfd.cGreenBits = 8; // g la kiti deu 
    pfd.cBlueBits = 8;  // b la 
    pfd.cAlphaBits = 8; // alpha la.. 

    // 2) get DC 
    ghdc = GetDC(ghwnd); /* pfd baddal genuin mahiti asnara DC */
    if(ghdc == NULL) 
    {
        fprintf(gpFile, "getDC() failed\n"); 
        return (-1); 
    }

    // 3) Get matching pixel format index using hdc and pfd 
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd); 
                                /*
                                    OS kade asnarya vivid pfd paiki mazya pfd la sarvaat jast match honara pfd cha
                                    index de 
                                */
    if(iPixelFormatIndex == 0) 
    {
        fprintf(gpFile, "ChoosePixelFormat() failed\n"); 
        return (-2); 
    }

    // 4) set the pixel format of found index. 
    if(SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
                            /* atapasson iPixelFormat la asnara pfd vapar dusra vapru nakos */
    {
        fprintf(gpFile, "SetPixelFormat() failed\n"); 
        return (-3); 
    } 

    /* Here our hdc is ready with openGL required pfd */

    // 5) Create rendering context using hdc, pfd and chosen pixel format index
    ghrc = wglCreateContext(ghdc); // this is not Win32 function. This is WGL (bridging API)  
            /* 
                we said we are creting context using 1)hdc 2)pfd and 3)chosen prixel format index 
                but here we are using only gdc 
                Because, pfd and pixel format index are already set in pfd in step-4
            */
    if(ghrc == NULL) 
    {
        fprintf(gpFile, "wglCreateContext() failed\n"); 
        return (-4); 
    }
    
    // 6) Make this rendering context as current context 
    // aataparyant ghdc current context hota, aata haa rendering context current context mhanun set kraycha aahe 
    // kaaran ata retained mode rendering kraych nahi aahe (je ghdc krto), RTR karaych aahe je ghrc krnar aahe..  
    if(wglMakeCurrent(ghdc, ghrc) == FALSE) 
    { 
        fprintf(gpFile, "wglMakeCurrent() failed\n"); 
        return (-5); 
    }
    
    // ***** FROM HERE ONWARDS OPENGL CODE STARTS *****  
    // tell OpenGL to choose color to clear the screen 
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f); /*  ERASEBKND kontya rangaani krach */

    return (0); 
}

void resize(int width, int height) 
{
    /* This function is called at least once as WM_RESIZE is posted at window creation */
    // code 
    // if height by accident becomes <=0, make height 1 
    if(height <= 0) 
    {
        height = 1; 
    }

    // Set the view port 
    glViewport(0, 0, (GLsizei)width, (GLsizei)height); 
                            /* 
                                openGL cha binocular window chya dimention shi map kr 
                            */
}

void display(void) 
{
    // code 
    // clear OpenGL buffers 
    glClear(GL_COLOR_BUFFER_BIT); 
    
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


/*
    steps : 
    1) initialize pixel format descriptor 
    2) get DC
    3) Get matching pixel format index using hdc and pfd 
    4) Set the pixel format of found index. 
    5) Create rendering context using hdc, pfd and PixelFormatIndex
    6) Make this rendering context as current context 
*/
