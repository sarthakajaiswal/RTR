#include <Windows.h> 
#include <stdio.h>      // file fopen(), fwrite() and fclose() 
#include <stdlib.h>     // exit() 
#include "Window.h" 

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
char gszLogFileName[] = "log.txt"; /* TCHAR not used because we here not using Win32 SDK, simple file IO can be done using char */
FILE *gpFile = NULL; 

// active window related variable 
BOOL gbActiveWindow = FALSE; 

// exit key pressed related 
BOOL gbEscapeKeyIsPressed = FALSE; 

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
                                                /*
                                                    OS GUI chya element sathi chi memory 3 prakare thevte ani vaprte 
                                                    .Fixed - DC sathi chi memory eka thikaani fix aste 
                                                    .Movable - memory location badlat aste 
                                                    .discarded - 

                                                    CS_OWNDC -> so this DC in fixed / reliable place, 
                                                                as this DC is going to be used for OpenGL                                                 
                                                */
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

        case WM_SIZE:
            resize(LOWORD(lParam), HIWORD(lParam)); 
            break; 

        case WM_KEYDOWN: 
            switch (wParam) 
            {
                case VK_ESCAPE: // vk - virtual key code 
                                // every key has virtual key code, but mainly used for keys not having character 
                    gbEscapeKeyIsPressed = TRUE; 
                    break; 

                default: 
                    break; 
            }
            break; 

        case WM_CHAR: 
            switch(wParam) 
            {
                /* 
                    this can be done in another way too.. 
                    ie. by calling toggleFullScreen(); in this message and 
                    let toggleFullScreen() analyze gbFullScreen 

                    But it is not done that way because, 
                    The reader after reading this code gets quick and clear information 
                    that screen is being toggled here.  
                */
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
                SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW); // go inside dwStyle -> dwStyle& 
                                                                        // and remove WS_OVERLAPPEDWND-> ~WS_OVERLAPPEDWINDOW) 

                SetWindowPos(           // 7 params 
                        ghwnd,          // 1-whichwindow
                        HWND_TOP,       // 2-keep window on top 3-left top
                        mi.rcMonitor.left, 
                        mi.rcMonitor.top, 
                        mi.rcMonitor.right - mi.rcMonitor.left, 
                        mi.rcMonitor.bottom - mi.rcMonitor.top, 
                        SWP_NOZORDER | SWP_FRAMECHANGED // tya window la kaahi message dyaycha ka - repaint kr, child dakhv, ... 
                ); 
                // NOZORDER = do not set the z-order as already set by us by saying HWND_TOP in param 2 
                // FRAMECHAGED = repaint kraych aahe 
                //               this internally posts WM_NCCALCSIZE message (nc-non client)  
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
            0, /* already set in SetWindowPlaceent so this 4 params kept zero */
            0, 
            0, 
            0, 
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED 
        ); 
        /*
            do not move - NOMOVE 
            do not resize - NOSIZE 
            do not allow your owner to chage z order - NOOWNERZORDER 
            do not yourself change z order - NOZORDER 
            repaint ke - FRAMECHANGED 
        */
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
    //  code 
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
        fprintf(gpFile, "Program terminated successfully\n"); 
        fclose(gpFile); 
        gpFile = NULL; 
    }
}
