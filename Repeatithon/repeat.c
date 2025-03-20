#include <Windows.h> 

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); 

int WINAPI WinMain(HISNTANce hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nShowCmd) 
{
    WNDCLASEX wnd; 
    MSG msg; 
    HWND hwnd; 

    static TCHAR szClassName[] = TEXT("The Standard Window"); 

    // window class initialization 
    wnd.cbSize = sizeof(WNDCLASSEX); 
    wnd.cbClsExtra = 0; 
    wnd.cbWndExtra = 0; 
    wmd.hIcon = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hIconSm = LoadIcon(NULL, IDI_APPLICATION); 
    wnd.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); 
    wnd.hCursor = LoadCursor(NULL, IDC_ARROW); 
    wnd.hInstance = hInstance; 
    wnd.lpszClassName = szclassName; 
    wnd.lpszMenuName = NULL; 
    wnd.lpfnWndProc = WndProc; 
    wnd.style = HREDRAW | VREDRAW; 

    // window Class registration 
    RegisterClassEx(&wnd); 

    // Create window 
    hwnd = CreateWindow(
        WS_OVERLAPPEDWINDOW, 
        szClassName, 
        TEXT("Sarthak Jaiswal"); 
    )
}