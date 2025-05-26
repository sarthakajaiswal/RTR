#include <Windows.h> 
#include "Window.h" 

// global function declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

// global variable declarations 
// variables related to fullscreen 
BOOL gbFullScreen = FALSE;  /* Hungarian Notation */ /* g-global b-bool */
                            /* Win32 SDK BOOL = TRUE(1) | FALSE (0) */
                            /* C++ BOOL = true(true) | false(false) */

HWND ghwnd = NULL; // making the variable global so that used across the functions 
                    // GLOBAL VARIABLE USE CASE 1 
                    // If function_1 calls function_2 then variable in function_1 can be used in function_2 
                    // using parameter mechanism 
                    // But if there is no direct call between them then make the variable global. 

DWORD dwStyle;  // can be defined local static too 
                // GLOBAL VARIABLE USE CASE 2 
                // make variable global so as to use it across the calls of re-enterent function. 

WINDOWPLACEMENT wpPrev; // length member must be initialized first to use WINDOWPLACEMENT variable 
                        // In C, initialization using (data manipulation statment) globally is not allowed. 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nShowCmd) 
{
    // variable declarations 
    static TCHAR szClassName[] = TEXT("The Standard Window"); 

    MSG msg; 
    WNDCLASSEX wnd; 
    HWND hwnd; 

    // code 
    ZeroMemory(&msg, sizeof(MSG)); 
    ZeroMemory(&wnd, sizeof(WNDCLASSEX)); 

    // window class initialization 
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wnd.style = CS_HREDRAW | CS_VREDRAW; 
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
        WS_EX_APPWINDOW,    // extended window style - ashi window dakhav je taskbar chya vrti yeil 
        szClassName, 
        TEXT("Sarthak Jaiswal"), 
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 
                    /*
                        CLIPCHILDREN | CLIPSIBLING => chidren ani siblings clip kr 
                        VISIBLE => clip kele tri mazi window visible asu de 
                    */
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
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

    // message loop 
    while(GetMessage(&msg, NULL, 0, 0)) 
    {
        TranslateMessage(&msg); 
        DispatchMessage(&msg); 
    }

    return ((int)msg.wParam); 
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) 
{
    // local function declarations 
    void toggleFullScreen(void); 

    // code 
    switch(iMsg) 
    {
        case WM_CREATE: 
            ZeroMemory((void*)&wpPrev, sizeof(WINDOWPLACEMENT)); // length member in wpPrev must be set before it is used 
                                                                // this setting is to be done once at starting 
                                                                // so initializing it in WWM_CREATE.  
            wpPrev.length = sizeof(WINDOWPLACEMENT); 
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

/* 
    ret_value fun_name(formal parameter list) 
    {
        empty body 
    }

    this skeleton is called as stub function 
*/