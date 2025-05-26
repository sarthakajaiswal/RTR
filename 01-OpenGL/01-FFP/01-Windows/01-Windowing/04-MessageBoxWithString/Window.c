// win32 headers 
#include <Windows.h> 

// global variable declarations 
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

// entry-point function 
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iShowCmd) 
{
    // variable declarations 
    WNDCLASSEX wndclass; 
    HWND hwnd; 
    MSG msg; 

    TCHAR szAppName[] = TEXT("RTR 6.0"); 

    // code 
    // window class initialization 
    wndclass.cbSize = sizeof(WNDCLASSEX); 
    wndclass.style = CS_HREDRAW | CS_VREDRAW; 
    wndclass.cbClsExtra = 0; 
    wndclass.cbWndExtra = 0; 
    wndclass.lpfnWndProc = WndProc; 
    wndclass.hInstance = hInstance; 
    wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); 
    wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION); 
    wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION); 
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wndclass.lpszClassName = szAppName; 
    wndclass.lpszMenuName = NULL; 

    // registration of window class 
    RegisterClassEx(&wndclass); 

    // create window 
    hwnd = CreateWindow(
                szAppName, 
                TEXT("Sarthak Ayodhyaprasad Jaiswal"), 
                WS_OVERLAPPEDWINDOW, 
                CW_USEDEFAULT, 
                CW_USEDEFAULT, 
                CW_USEDEFAULT, 
                CW_USEDEFAULT, 
                NULL, 
                NULL, 
                hInstance, 
                NULL 
            );
            
    // show window 
    ShowWindow(hwnd, iShowCmd); 

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
    // variable declarations 
    int a = 10; 
    int b = 20; 
    int sum; 

    char str[255]; 

    // code 
    sum = a + b; 
    wsprintf(str, "The sum of %d and %d is %d", a, b, sum); 

    switch(uMsg) 
    {
        case WM_CREATE: 
            MessageBox(hwnd, TEXT(str), TEXT("WM_CRETE"), MB_OK); 
            break; 

        case WM_SIZE: 
            break; 

        case WM_MOVE: 
            break; 

        case WM_KEYDOWN: 
            switch(wParam) 
            {
                case VK_ESCAPE: 
                    break; 
                
                default: 
                    break; 
            }
            break; 

        case WM_CHAR: 
            switch(wParam) 
            {
                case 'f': 
                    break; 

                case 'F': 
                    break; 

                default: 
                    break; 
            }
            break; 

        case WM_LBUTTONDOWN: 
            break; 

        case WM_CLOSE: 
            DestroyWindow(hwnd); 
            break; 

        case WM_DESTROY: 
            PostQuitMessage(0); 
            break; 

        default: 
            break; 
    }

    return (DefWindowProc(hwnd, uMsg, wParam, lParam)); 
}
