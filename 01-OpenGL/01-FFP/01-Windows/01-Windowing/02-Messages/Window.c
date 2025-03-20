#include <Windows.h> 

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nShowCmd) 
{
    // variable declarations 
    static TCHAR szClassName[] = TEXT("The Standard Window"); 
    
    HWND hwnd; 
    MSG msg; 
    WNDCLASSEX wnd; 

    // code 
    ZeroMemory(&msg, sizeof(MSG)); 
    ZeroMemory(&wnd, sizeof(WNDCLASSEX)); 

    // initialization of window class 
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

    // registration of window class 
    RegisterClassEx(&wnd); 

    // create window 
    hwnd = CreateWindow(
        szClassName, 
        TEXT("Sarthak Jaiswal"), 
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
    switch(uMsg) 
    {
        case WM_CREATE: 
            MessageBox(hwnd, TEXT("This is the first Message"), TEXT("WM_CREATE"), MB_OK); 
            break; 

        case WM_SIZE: 
            MessageBox(hwnd, TEXT("Window size changed"), TEXT("WM_SIZE"), MB_OK); 
            break; 

        case WM_MOVE: 
            MessageBox(hwnd, TEXT("Window is moved"), TEXT("WM_MOVE"), MB_OK); 
            break; 

        case WM_KEYDOWN: 
            switch(wParam) 
            {
                case VK_ESCAPE: 
                    MessageBox(hwnd, TEXT("Escape key is pressed"), TEXT("WM_KEYDOWN"), MB_OK); 
                    break; 

                default: 
                    break; 
            }
            break; 

        case WM_CHAR: 
            switch(wParam) 
            {
                case 'F': 
                    MessageBox(hwnd, TEXT("'F' is pressed"), TEXT("WM_CHAR"), MB_OK); 
                    break; 

                case 'f': 
                    MessageBox(hwnd, TEXT("'f' is pressed"), TEXT("WM_CHAR"), MB_OK); 
                    break; 

                default: 
                    break; 
            }
            break; 

        case WM_LBUTTONDOWN: 
            MessageBox(hwnd, TEXT("Left mouse button is clicked"), TEXT("WM_LBUTTONDOWN"), MB_OK); 
            break; 

        case WM_CLOSE: 
            MessageBox(hwnd, TEXT("Window is closing"), TEXT("WM_CLOSE"), MB_OK); 
            DestroyWindow(hwnd); 
            break; 

        // convenctionally, WM_DESTROY shoulf contain PostQuitMessage() only 
        // so to do uninitialization if any, (eg- freeig memory) 
        // WM_CLOSE is handelled and unitialize() is called inside it. 

        case WM_DESTROY: 
            PostQuitMessage(0); 
            break; 

        default: 
            break; 
    }

    // देवानी, जे काम करण्यासाठी पाठवलं ते काम करायची ( handle that messages only that are intended to) 
    // आणि बाकी ईश्वरावर (DefWindowProc()) सोडाचं . 
    
    // जे काम आपन केलं ते सुद्धा बघतो देव. (control flow reaches to DefWindowProc either after handeling or without it) 
    return (DefWindowProc(hwnd, uMsg, wParam, lParam)); 
} 
