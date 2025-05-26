// Win32 headers
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Custom heander files
#include "OGL.h"

// OpenGL header files
#include <gl/GL.h>
#include <gl/GLU.h>

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
HDC ghdc = NULL; //HDC - Handle to Device Context
HGLRC ghrc = NULL; // HGLRC - Handle to Graphics Library Rendering Context

// variables related to File I/O
char gszLogFileName[] = "Log.txt";
FILE *gpFile = NULL;

// Active Window related variables
BOOL gbActiveWindow = FALSE;

// Exit keypress related
BOOL gbEscapeKeyIsPressed = FALSE;

// rotation variables 
GLfloat anglePyramid = 0.0f; 
GLfloat angleCube = 0.0f;

// texture related global variables 
GLuint texture_stone; 
GLuint texture_kundali; 

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

	// *** Registration of Window Class ***
	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
						  szAppName,
						  TEXT("Sarthak Ayodhyaprasad Jaiswal"),
						  WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						  CW_USEDEFAULT, 
						  CW_USEDEFAULT, 
						  WIN_WIDTH,	 
						  WIN_HEIGHT,	 
						  NULL,
						  NULL,
						  hInstance,
						  NULL);
	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);

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
	// function declaration/prototypes
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
		case VK_ESCAPE: // VK - Virtual Keycode
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
			}
		}

		ShowCursor(FALSE); 
	}
	else
	{
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowLong(ghwnd, GWL_STYLE, dwstyle | WS_OVERLAPPED); // '|' adds style , '~' removes style
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		
		ShowCursor(TRUE);
	}
}

int initialize(void)
{
    // local function declarations 
	void resize(int, int);
	BOOL loadGLTexture(GLuint*, TCHAR[]); 

	// variable declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex = 0;

	// code
	// Pixel Format Descriptor initialisation
	ZeroMemory((void *)&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1; 
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
    pfd.cDepthBits = 32; // in mobile this value is 24, should be multiple of 8 and <= cColorBits 
	
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
    // depath related code 
    glShadeModel(GL_SMOOTH); 
    glClearDepth(1.0f);     
    glEnable(GL_DEPTH_TEST); 
    glDepthFunc(GL_LEQUAL); 
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); 
                            

	// Instruct OpenGl to choose the colour to clear the screen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// load textures 
	if(loadGLTexture(&texture_stone, MAKEINTRESOURCE(IDBITMAP_STONE)) == FALSE) 
	{
		fprintf(gpFile, "loadGLTexture() failed to create stone texture\n"); 
		return (-6); 
	}

	if(loadGLTexture(&texture_kundali, MAKEINTRESOURCE(IDBITMAP_KUNDALI)) == FALSE) 
	{
		fprintf(gpFile, "loadGLTexture() failed to create kundali texture\n"); 
		return (-7); 
	}

	// enabling texturing 
	glEnable(GL_TEXTURE_2D); 

	// Warmup resize
	resize(WIN_WIDTH, WIN_HEIGHT);
		
	return (0);
}

BOOL loadGLTexture(GLuint* texture, TCHAR imageResourceID[]) 
{
	// variable declarations 
	HBITMAP hBitmap = NULL; 
	BITMAP bmp; 
	BOOL bResult = FALSE; 

	// code 
	// load the bitmap as image 	
	hBitmap = (HBITMAP)LoadImage(			/* value is typecasted accordingly this function can load various types of images, here we are loading bitmap so return */ 
					GetModuleHandle(NULL), 	/* hInstance, passing NULL to this function return handle to current instance */
					imageResourceID, 
					IMAGE_BITMAP, 			/* je image load krachi aahe tyacha type */
					0, 0,  					/* width and height pf image width and height is given ONLY when the image is of icon or cursor */
					LR_CREATEDIBSECTION 	/* LR -> load resource, DIB -> device independent bitmap | mazya image cha dib tayar krun load kr */
				); 
	if(hBitmap)
	{
		bResult = TRUE; 

		// get bitmap structure from the loaded bitmap image 
		GetObject(hBitmap, sizeof(BITMAP), &bmp); 

		// generate OpenGL texture object 
		glGenTextures(1, texture); // ya function ne rikama texture tayar hoto 
				// 1 -> kiti texture generate krache aahe 
				// texture -> generate kelela texture kuthe thewaycha  

		// bind to newly created empty texture object 
		glBindTexture(
			GL_TEXTURE_2D, 	/* bind kuth kru */
			*texture		/* bind kunala kru */
		); 

		// unpack the image in memory for faster loading 
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4); 	/* 1 -> memory madhe image unpack kr ani alignment krun thev */
												/* 2 -> 4 ne alignment kr */
		// texture ks pahije he sangane
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
												/* 
													glTexParameteri -> texture che parameter set krne 
													gl_Texture_2d -> jya texture baddal bolto aahe to texture kute bind kela aahe 
													mag -> object magnification/jawal asel tevha cha texture 
													gl_linear -> high quality cha de 
												*/ 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
												/*
													min -> minification
													gl_linear_mipmap_linear -> mipmapping sathi garjecha aahe tevdhach linear kr  
												*/
		gluBuild2DMipmaps(GL_TEXTURE_2D, 3, bmp.bmWidth, bmp.bmHeight, GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits); 
							/*
								1 -> texture cha object kuth bind kela aahe kuth 
								2 -> texture cha format kaay aahe -> 3 kivha RGB 
								3 -> bmp image chi height 
								4 -> bmp image chi width 
								5 -> 7vya parameter madhlya data cha format, (GL_RGB dil tr image ulti dyavi lagte) 
								6 -> 7vya parameter madhlya data cha type 
								7 -> actual texture cha data  
							*/

		glBindTexture(GL_TEXTURE_2D, 0); 
		DeleteObject(hBitmap); 
		hBitmap = NULL; 
	}

	// gen - bind - unbind triplet 

	return (bResult); 
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
	gluPerspective(45.0f, 
				  (GLfloat)width/(GLfloat)height, 
				   0.1f, 
				   100.0f 
				   );	
}

void display(void)
{
	// code
	// Clear OpenGL buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.5f, 0.0f, -6.0f);
	glRotatef(anglePyramid, 0.0f, 1.0f, 0.0f); 
	glBindTexture(GL_TEXTURE_2D, texture_stone); 

	// pyramid drawing code
	glBegin(GL_TRIANGLES);

    // front face 
	glTexCoord2f(0.5f, 1.0f); 
	glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, 1.0f);	
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(1.0f, -1.0f, 1.0f); 
    
    // right face 
	glTexCoord2f(0.5f, 1.0f); 
    glVertex3f(0.0f, 1.0f, 0.0f); 
	glTexCoord2f(0.0f, 0.0f); 
    glVertex3f(1.0f, -1.0f, 1.0f); 
	glTexCoord2f(1.0f, 0.0f); 
    glVertex3f(1.0f, -1.0f, -1.0f); 
    
    // back face 
	glTexCoord2f(0.5f, 1.0f); 
    glVertex3f(0.0f, 1.0f, 0.0f); 
	glTexCoord2f(1.0f, 0.0f); 
    glVertex3f(1.0f, -1.0f, -1.0f); 
	glTexCoord2f(0.0f, 0.0f); 
    glVertex3f(-1.0f, -1.0f, -1.0f); 
    
    // left face 
	glTexCoord2f(0.5f, 1.0f); 
    glVertex3f(0.0f, 1.0f, 0.0f); 
	glTexCoord2f(0.0f, 0.0f); 
    glVertex3f(-1.0f, -1.0f, -1.0f);  
	glTexCoord2f(1.0f, 0.0f); 
    glVertex3f(-1.0f, -1.0f, 1.0f); 

	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0); 
	
	// cube drawing code 
	glLoadIdentity(); 

	glTranslatef(1.5f, 0.0f, -6.0f);
	glRotatef(angleCube, 1.0f, 0.0f, 0.0f); 
	glRotatef(angleCube, 0.0f, 1.0f, 0.0f); 
	glRotatef(angleCube, 0.0f, 0.0f, 1.0f); 
	// glRotatef(angleCube, 1.0f, 1.0f, 1.0f); 
	glScalef(0.75f, 0.75f, 0.75f); 

	glBindTexture(GL_TEXTURE_2D, texture_kundali); 

	// cube drawing code
	glBegin(GL_QUADS); 

	// front face /* front-> 1 */ 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(1.0f, 1.0f, 1.0f); // right top /* right->1, top->1 */ 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-1.0f, 1.0f, 1.0f);	// left top /* left->-1, top->1 */ 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, 1.0f);  // left bottom /* left->-1, bottom->-1 */ 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(1.0f, -1.0f, 1.0f); // right bottom /* right->1, bottom->-1 */ 
	
	// right face 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(1.0f, 1.0f, -1.0f); 	// right top 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(1.0f, 1.0f, 1.0f);	// left top 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(1.0f, -1.0f, 1.0f);	// left bottom 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(1.0f, -1.0f, -1.0f); // right bottom  
	
	// back face 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(-1.0f, 1.0f, -1.0f);	 // right top 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(1.0f, 1.0f, -1.0f);	 // left top 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(1.0, -1.0, -1.0f); 	 // left bottom 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, -1.0f); // right bottom  
	
	// left face 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(-1.0f, 1.0f, 1.0f);	// right top 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-1.0f, 1.0f, -1.0f);// left top 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, -1.0f); // left bottom 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, 1.0f); // right bottom 

	// top face 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(1.0f, 1.0f, -1.0f); 	// right top 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-1.0f, 1.0f, -1.0f);	// left top 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-1.0f, 1.0f, 1.0f);	// left bottom 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(1.0f, 1.0f, 1.0f);   // right bottom 
	
	// bottom face 
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(1.0f, -1.0f, 1.0f);	// right top 
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-1.0f, -1.0f, 1.0f); // left top 
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-1.0f, -1.0f, -1.0f);	// left bottom 
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(1.0f, -1.0f, -1.0f); 	// right bottom 
	
	glEnd(); 

	glBindTexture(GL_TEXTURE_2D, 0); 
	
	// Swap the buffers
	SwapBuffers(ghdc);
}

void update(void)
{
	// code
	anglePyramid = anglePyramid + 0.02f; 

	if(anglePyramid >= 360.0f) 
	{
		anglePyramid = anglePyramid - 360.0f; 
	}

	angleCube = angleCube + 0.02f;

	if (angleCube >= 360.0f)
	{
		angleCube = angleCube - 360.0f;
	}
}

void uninitialize(void)
{
	
	// function declarations
	void toggleFullScreen(void);
		
	// code
	if ( gbFullScreen == TRUE)
	{
		toggleFullScreen();
	}
	gbFullScreen = FALSE;

	if(texture_kundali) 
	{
		glDeleteTextures(1, &texture_kundali); 
		texture_kundali = 0;  
	}

	if(texture_stone) 
	{
		glDeleteTextures(1, &texture_stone); 
		texture_stone = 0; 
	}

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

	// close the file
	if (gpFile)
	{
		fprintf(gpFile, "Programme Terminated Successfully.");
		fclose(gpFile);
		gpFile = NULL;
	}
}
