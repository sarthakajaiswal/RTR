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
    ZeroMemory(&wnd, sizeof(WNDCLASSEX)); 
    ZeroMemory(&msg, sizeof(MSG)); 

    // initialization of window class 
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wnd.style = CS_HREDRAW | CS_VREDRAW; 
    wnd.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH); 
    wnd.hIcon = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hIconSm = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wnd.hInstance = hInstance; 
    wnd.lpfnWndProc = WndProc; 
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
    // variable declarations 
    HDC hdc; 
    PAINTSTRUCT ps; 
    RECT rc; 
    TCHAR str[] = TEXT("Hello World!!!"); 

    static int iPaintFlag = -1;     // made static so that it can be used across the messages 

    // code 
    switch(uMsg) 
    {
        case WM_PAINT: 
            GetClientRect(hwnd, &rc); // empty rc is filled with client area of rc 
            // 1) कोणत्या rectangle वर print कराच त्याचे dimention मिळाले  

            // hdc = GetDC(hwnd); // hdc - handle to device context 
                                  // context - अवस्था  
            // डिव्हाईस ची अवस्था सांगणार्या structure चा handle
            // here device is Graphics Card 

            hdc = BeginPaint(hwnd, &ps); 
            // 2) Paint करणारा विशेषज्ञ मिळाला  
            // BeginPaint() ने मिळालेला विशेषज्ञ WM_PAINT मधे paint करण्यासाठीची genuin माहिती ठेवतो. 
            
            SetBkColor(hdc, RGB(0, 0, 0)); // param1 -> background चा रंग set करणारा specialist - hdc
            // param2 -> कोणता रंग? 
            // 3) background रंग दिला 

            if(iPaintFlag == 1) 
                SetTextColor(hdc, RGB(255, 0, 0 )); 
            else if (iPaintFlag == 2) 
                SetTextColor(hdc, RGB(0, 255, 0)); 
            else if (iPaintFlag == 3) 
                SetTextColor(hdc, RGB(0, 0, 255)); 
            else if (iPaintFlag == 4) 
                SetTextColor(hdc, RGB(255, 255, 0)); 
            else if (iPaintFlag == 5) 
                SetTextColor(hdc, RGB(0, 255, 255)); 
            else if (iPaintFlag == 6) 
                SetTextColor(hdc, RGB(255, 0, 255)); 
            else if(iPaintFlag == 7) 
                SetTextColor(hdc, RGB(255, 255, 255)); 
            // 4) text रंग दिला  

            DrawText(hdc, str, -1, &rc, DT_SINGLELINE | DT_CENTER | DT_VCENTER); 
            // 5) print करायचा text

            EndPaint(hwnd, &ps); 
            // ReleaseDC(hwnd, hdc); 
            break; 

        case WM_CHAR: 
            switch(wParam) 
            {
                case 'R': 
                case 'r': 
                    iPaintFlag = 1; 
                    InvalidateRect(hwnd, NULL, TRUE);  // forcefully posting WM_PAINT 
                    break; 

                case 'G': 
                case 'g': 
                    iPaintFlag = 2; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 

                case 'B': 
                case 'b': 
                    iPaintFlag = 3; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 

                case 'Y': 
                case 'y': 
                    iPaintFlag = 4; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 

                case 'C': 
                case 'c': 
                    iPaintFlag = 5; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 

                case 'M': 
                case 'm': 
                    iPaintFlag = 6; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 

                case 'W': 
                case 'w': 
                    iPaintFlag = 7; 
                    InvalidateRect(hwnd, NULL, TRUE); 
                    break; 
            }
            break; 

        case WM_DESTROY: 
            PostQuitMessage(0); 
            break; 

        default: 
            break; 
    }

    return (DefWindowProc(hwnd, uMsg, wParam, lParam)); 
}
