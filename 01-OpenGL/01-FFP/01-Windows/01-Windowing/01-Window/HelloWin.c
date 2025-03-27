#include <Windows.h> 

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

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
    wnd.hIcon = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hIconSm = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wnd.lpfnWndProc = WndProc; 
    wnd.hInstance = hInstance; 
    wnd.lpszClassName = szClassName; 
    wnd.lpszMenuName = NULL; 

    // window class registration 
    RegisterClassEx(&wnd); 

    // create window from registered class 
    hwnd = CreateWindow(
        szClassName, 
        TEXT("Sarthak Jaiswal"), 
        WS_OVERLAPPEDWINDOW, 
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
        CW_USEDEFAULT, 
        NULL,   // Jiska koi nahi uska to khuda hai yaaro 
        NULL, 
        hInstance, 
        NULL
    ); 

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

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) 
{
    /*
        . 
        pre-hooking code lies here  
        . 
    */

    switch(uMsg) 
    {
        case WM_DESTROY: 
            PostQuitMessage(0); 
            break; 

        default: 
            break; 
    }

    /*
        . 
        Post hooking code lies here 
        Post Hooking is mainly used for Hacking 
        . 
    */

    return (DefWindowProc(hwnd, uMsg, wParam, lParam)); 
}
