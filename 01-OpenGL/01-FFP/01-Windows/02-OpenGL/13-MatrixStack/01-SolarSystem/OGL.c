// Win32 headers
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Custom heander files
#include "OGL.h"

// OpenGL header files
#include <gl/GL.h>
#include <gl/GLU.h>	/* needed for spehere too */

//OpenGL libraries
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

// Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

// global function declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// global variable declarations
BOOL gbFullScreen = FALSE;
HWND ghwnd = NULL;
DWORD dwstyle;
WINDOWPLACEMENT wpPrev;

//OpenGL related variables
HDC ghdc = NULL; 
HGLRC ghrc = NULL; 

// variables related to File I/O
char gszLogFileName[] = "Log.txt";
FILE *gpFile = NULL;

// Active Window related variables
BOOL gbActiveWindow = FALSE;

// Exit keypress related
BOOL gbEscapeKeyIsPressed = FALSE;

// sloar system related variables 
int year = 0; 
int date = 0; 
GLUquadric *quadric = NULL; 

// Entry-point function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevinstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// Function declarations
	int initialize(void);
	void resize(int, int);

	void display(void);
	void update(void);
	void uninitialize(void);

	// variable declarations
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("RTR6");
	BOOL bDone = FALSE;

	// code
	// Create Log file
	gpFile = fopen(gszLogFileName, "w");
	if (gpFile == NULL)
	{
		MessageBox(NULL, TEXT("Log file creation FAILED!!!"), TEXT("File I/O error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Programme Started Successfully!\n");
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; 
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
						  szAppName,
						  TEXT("Sarthak Ayodhyaprasad Jaiswal"),
						  WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						  CW_USEDEFAULT, // X co-ordintate of window
						  CW_USEDEFAULT, // Y co-ordintate of window
						  WIN_WIDTH,	 // Width of window
						  WIN_HEIGHT,	 // Height of window
						  NULL,
						  NULL,
						  hInstance,
						  NULL);

	ghwnd = hwnd;

	// show window
	ShowWindow(hwnd, iCmdShow);

	// Paint background of window
	UpdateWindow(hwnd);

	// Initialize
	int result = initialize();
	if (result != 0)
	{
		fprintf(gpFile, "Initialize() FAILED! \n");
		DestroyWindow(hwnd);
		hwnd = NULL;
	}
	else
	{
		fprintf(gpFile, "Initialize() Completed Successfully.\n");
	}

	// Set this window as Foreground & Active window
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	// Game Loop
	while (bDone == FALSE)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) 
		{
			if (msg.message == WM_QUIT)
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
			if (gbActiveWindow == TRUE)
			{
				if (gbEscapeKeyIsPressed == TRUE)
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

	// Uninitialize
	uninitialize();

	return ((int)msg.wParam);
}

// Callback function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// function declaration 
	void toggleFullScreen(void);

	// code
	switch (iMsg)
	{

	case WM_CREATE:
		ZeroMemory((void *)&wpPrev, sizeof(WINDOWPLACEMENT)); 
		wpPrev.length = sizeof(WINDOWPLACEMENT);			  

		break;

	case WM_SETFOCUS:
		gbActiveWindow = TRUE;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = FALSE;
		break;

	case WM_ERASEBKGND:	
		return(0);
	
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
		switch (wParam)
		{
		case 'F':
		case 'f':
			if (gbFullScreen == FALSE)
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
		
		case 'D': 
			date = (date + 6)%360; /* remainder is not required to float, so that taken as 'int'*/ 
									/* 360 degree ch ek revolution tr 'date' degree ch kiti he kadhnyasathi ch ganit */
			break; 
		
		case 'd': 
			date = (date - 6)%360; 
			break; 

		case 'Y': 
			year = (year + 3) % 360; 
			break; 

		case 'y': 
			year = (year - 3) % 360; 
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

void toggleFullScreen(void)
{
	// variable declarations
	MONITORINFO mi;

	// code
	if (gbFullScreen == FALSE)
	{
		dwstyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwstyle & WS_OVERLAPPEDWINDOW)
		{
			ZeroMemory((void *)&mi, sizeof(MONITORINFO));
			mi.cbSize = sizeof(MONITORINFO);

			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwstyle & ~WS_OVERLAPPEDWINDOW); // '~' removes or negates value
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);

				/*
				SWP_NOZORDER(SWP - Set Window Postion) -> do not set Z order, we have set to top using HWND_TOP

				SWP_FRAMECHANGED sends a message internally calls WM_NCCALSIZE(Window Message Non Client Area Calculate Sizes)
					Client Area is the window part between borders & caption bar. Non-Client Area is the caption bar containing system menu & min/max boxes.
				*/
			}
		}

		ShowCursor(FALSE); // Optional step : If upon Full Screen, should the cursor be displayed
	}
	else
	{
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowLong(ghwnd, GWL_STYLE, dwstyle | WS_OVERLAPPED); // '|' adds style , '~' removes style
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		/*
		SWP_NOMOVE - Do not move as SetWindowPlacement as already set the position
		SWP_NOSIZE - Do not resize
		SWP_NOOWNERZORDER - Do not let Owner make changes
		*/

		ShowCursor(TRUE);
	}
}

int initialize(void)
{
	// variable declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex = 0;

	void resize(int, int);

		
	// code
	// Pixel Format Descriptor initialisation
	ZeroMemory((void *)&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1; // FFP in Windows, expects this to be 1 by default
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32; 
	
	// getdc
	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() function FAILED! \n");
		return(-1);
	}

	// Get matching Pixel Format Index using hdc & pfd
	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() FAILED! \n");
		return(-2);
	}

	// Select the Pixel Format of found index
	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() FAILED! \n");
		return(-3);
	}

	// Create Rendering Context using hdc, pfd & iPixelFormatIndex
	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() FAILED! \n");
		return(-4);
	}

	// Make this rendering context as current context
	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() FAILED! \n");
		return(-5);
	}

	// OpenGL code starts here ...
	// depth related code 
	glShadeModel(GL_SMOOTH); 
	glClearDepth(1.0f); 
	glEnable(GL_DEPTH_TEST); 
	glDepthFunc(GL_LEQUAL); 
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); 

	// Instruct OpenGl to choose the colour to clear the screen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// initialize quadric 
	quadric = gluNewQuadric(); 
	
	// Warmup resize
	resize(WIN_WIDTH, WIN_HEIGHT);
		
	return (0);
}

void resize(int width, int height)
{
	// code
	// If height accidentally becomes 0 or less, then make it 1
	if(height <= 0)
	{
		height = 1;
	}

	// Set the View port
	glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));

	// Set Matrix Project Mode
	glMatrixMode(GL_PROJECTION);

	// Set to Identity Matrix
	glLoadIdentity();

	//  pective
	gluPerspective(45.0f, // Field of View
				  (GLfloat)width/(GLfloat)height, // Aspect ratio
				   0.1f, //Near
				   100.0f // Far
				   );	

}

void display(void)
{
	// code
	// Clear OpenGL buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set Matrix to Model View mode
	glMatrixMode(GL_MODELVIEW);
	
	// Set to Identity Matrix
	glLoadIdentity();
	
	// do view transformation 
	gluLookAt(
		0.0f, 0.0f, 5.0f, 
		0.0f, 0.0f, 0.0f, 
		0.0f, 1.0f, 0.0f 
	); 

	// code for sun 
	// save above transformation into modelview matrix stack 
	glPushMatrix(); 

	// adust the poles of sphere of sun 
	glRotatef(90.0f, 1.0f, 0.0f, 0.0f); 

	// set sun's polygon properties 
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); /* polygon che front ani back donhi consider kr */ 

	// make sun yellow 
	glColor3f(1.0f, 1.0f, 0.0f); 

	gluSphere(quadric, 0.75f, 30.0f, 30.0f); /* 30.0, 30.0 -> horizontal and vertical subdivisions */ 
											 /* more the subdivisions smoother the surface */
											 /* also called as segmentation, tissilation */
	
	// after drawing the sphere we need to go back to original to draw earth 
	// hence restore the matrix 
	glPopMatrix(); 

	// code for earth 
	// again save this matrix to let us to come back again 
	glPushMatrix(); 

	// let the earth allow to revolve around the sun 
	glRotatef((GLfloat)year, 0.0f, 1.0f, 0.0f); 

	// earth is at some distance from sun 
	glTranslatef(1.5f, 0.0f, 0.0f); 

	// adjust poles of sphere of earth 
	glRotatef(90.0f, 1.0f, 0.0f, 0.0f); 

	// now let the earth spin around itself 
	glRotatef((GLfloat)date, 0.0f, 0.0f, 1.0f); // here z rotation is done as gluSphere's z axis is vertical now 
	
	// set earth's polygon properties 
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 

	// make earth bluish 
	glColor3f(0.4f, 0.7f, 1.0f); 

	// draw sphere of earth 
	gluSphere(quadric, 0.2f, 20, 20); 

	// to go back to origin 
	glPopMatrix(); 

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

	// If iser is exiting FullScreen, then restore screen/window

	if ( gbFullScreen == TRUE)
	{
		toggleFullScreen();
	}
	gbFullScreen = FALSE;

	// Make hdc as current context by releasing rendering context as current context
	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);

	}

	// Delete the rendering context
	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	// Relese the DC
	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	// Destroy window
	if (ghwnd)
	{
		DestroyWindow(ghwnd);
		ghwnd = NULL;
	}

	// delete quadric 
	if(quadric) 
	{
		gluDeleteQuadric(quadric); 
		quadric = NULL; 
	}

	// close the file
	if (gpFile)
	{
		fprintf(gpFile, "Programme Terminated Successfully.");
		fclose(gpFile);
		gpFile = NULL;
	}
}
