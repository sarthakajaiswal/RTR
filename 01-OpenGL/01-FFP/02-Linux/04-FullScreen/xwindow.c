// standard header files
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

// Xlib head files
#include <X11/Xlib.h>
#include <X11/Xutil.h> // for visualInfo and related APIs
#include <X11/XKBlib.h> // keyboard related Xlib APIs

// macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

// global variables
Display* gpDisplay = NULL;
XVisualInfo visualInfo;
Window window;
Colormap colorMap;

// full-screen related variables
Bool bFullScreen = False;

// entry-point function
int main(void)
{
    // function declarations
    void uninitialize(void);
    void toggleFullScreen(void);

    // variable declarations
    int defaultScreen;
    int defaultDepth;
    Status status;
    XSetWindowAttributes windowAttributes;
    Atom windowManagerDeleteAtom;
    XEvent event;
    Screen* screen = NULL;
    int screenWidth, screenHeight;
    KeySym keySym; // sym->symbol
    char keys[26]; // although we need only 0 index, conventionally array size is equal to number of characters either small(26), capital(26) or both(52)

    // code
    // 1) open connection with xserver
    gpDisplay = XOpenDisplay(NULL);
    if(gpDisplay == NULL)
    {
        printf("XOpenDisplay() failed to connect with the server\n");
        uninitialize();
        exit(1);
    }

    // 2) create the default screen object
    defaultScreen = XDefaultScreen(gpDisplay);

    // 3) get default depth
    defaultDepth = XDefaultDepth(gpDisplay, defaultScreen);

    // get visual info
    memset((void*)&visualInfo, 0, sizeof(XVisualInfo));
    status = XMatchVisualInfo(gpDisplay, defaultScreen, defaultDepth, TrueColor, &visualInfo);
    if(status == 0)
    {
        printf("XMatchVisualInfo() failed\n");
        uninitialize();
        exit(1);
    }

    // set window attributes
    memset((void*)&windowAttributes, 0, sizeof(XSetWindowAttributes));
    windowAttributes.border_pixel = 0; 
    windowAttributes.background_pixmap = 0;
    windowAttributes.background_pixel = XBlackPixel(gpDisplay, visualInfo.screen);
    windowAttributes.colormap = XCreateColormap(
				    gpDisplay, 
				    XRootWindow(gpDisplay, visualInfo.screen), 
				    visualInfo.visual, 
				    AllocNone
			    );  
    // This is one of giving mask             /* another way for giving event mask -> XSelectInput() */
    windowAttributes.event_mask = KeyPressMask | ButtonPressMask | FocusChangeMask | StructureNotifyMask | ExposureMask;
                        /* WM_SIZE      -> StructureNotify
                         * WM_KEYDOWN   -> KeyPress
                         * WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN -> ButtonPress
                         * WM_SETFOCUS  -> FocusIn
                         * WM_KILLFOCUS -> FocusOut
                         * WM_PAINT     -> Expose
                         * WM_DESTROY   -> DestroyNotify (WM dependent), 33(WM neutral, because we have created ATOM for destroy event)
                         * WM_MOUSEMOVE -> PointerMotionMask
                         * Hide/Show Window -> VisibilityChangeMask
                        */
                        /* some events are sent irrespective of event masking like MapNotify(analogous to WM_CREATE) */

    colorMap = windowAttributes.colormap;

    window = XCreateWindow(
	    gpDisplay, 
	    XRootWindow(gpDisplay, visualInfo.screen), // root window
	    0, //x
	    0, // y
	    WIN_WIDTH, 
	    WIN_HEIGHT, 
	    0, // border width -> default 
	    visualInfo.depth, 
	    InputOutput, // window ksahsathi pahije input/output/donhi
	    visualInfo.visual, // gc cha structure
	    CWBorderPixel | CWBackPixel | CWEventMask | CWColormap, // styles
	    &windowAttributes 
    ); 

    if(!window) 
    {
	    printf("XCreateWindowFailed\n"); 
	    uninitialize(); 
	    exit(EXIT_FAILURE); 
    } 

    windowManagerDeleteAtom = XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(gpDisplay, window, &windowManagerDeleteAtom, 1); 

    // set window title 
    XStoreName(gpDisplay, window, "SAJ:XWindow"); 

    // map the window to show it 
    XMapWindow(gpDisplay, window); 

    // centering of window
    screen = XScreenOfDisplay(gpDisplay, visualInfo.screen);
    screenWidth = XWidthOfScreen(screen);
    screenHeight = XHeightOfScreen(screen);
    XMoveWindow(gpDisplay, window, screenWidth/2-WIN_WIDTH/2, screenHeight/2-WIN_HEIGHT/2);
    XMoveWindow(gpDisplay, window, screenWidth/2-WIN_WIDTH/2, screenHeight/2-WIN_HEIGHT/2);

    // message loop 
    while(1) 
    {
	    XNextEvent(gpDisplay, &event);
	    switch(event.type) 
	    {
            case MapNotify:
                break;

            case FocusIn:
                break;

            case FocusOut:
                break;

            case ConfigureNotify:
                break;

            case KeyPress:
                // for escape key
                keySym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0); // 3rd 0=> if key combination is used 4th=> shift is used in key combination
                switch(keySym)
                {
                    case XK_Escape:
                        uninitialize();
                        exit(EXIT_SUCCESS);
                        break;
                    default:
                        break;
                }

                // for alphabetic key press
                XLookupString(&event.xkey, keys, sizeof(keys), NULL, NULL);  // keypress madhe kahi string aalya astil tr lookup kr
                                // 4th = to save state of xlookupstring if used across multiple message
                                // 5th = given if the state is used for further propogration of events
                switch(keys[0])
                {
                    case 'F':
                    case 'f':
                        if(bFullScreen == False)
                        {
                            toggleFullScreen();
                            bFullScreen = True;
                        }
                        else
                        {
                            toggleFullScreen();
                            bFullScreen = False;
                        }
                        break;

                    default:
                        break;
                }
                break;

            case ButtonPress:
                break;

            case Expose:
                break;

		    case 33:
			    uninitialize(); 
			    exit(EXIT_SUCCESS); 
			    break; 
			    
		    default: 
			    break; 
	    } 

    } 

    uninitialize(); 
    return (0); 
} 

void toggleFullScreen(void)
{
    // code
    Atom windowManagerNormalStateAtom = XInternAtom(gpDisplay, "_NET_WM_STATE", False); // _NET -> this atom is network complient
    Atom windowManagerFullScreenStateAtom = XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);

    XEvent event;
    memset((void*)&event, 0, sizeof(XEvent));
    event.type = ClientMessage;     // event cha type kuthla aahe?
    event.xclient.window = window;  // kontya window sathi janaar aahe?
    event.xclient.message_type = windowManagerNormalStateAtom;
    event.xclient.format = 32;
    event.xclient.data.l[0] = bFullScreen ? 0 : 1;
    event.xclient.data.l[1] = windowManagerFullScreenStateAtom;

    // send above event to XServer
    // till now we have used APIs to send messages to XServer, this is first time we are creating a event as per our requirement and sending it to server
    XSendEvent(
        gpDisplay,
        XRootWindow(gpDisplay, visualInfo.screen),
        False,
        SubstructureNotifyMask,
        &event
    );
}

void uninitialize(void) 
{
	// code 
	if(window) 
	{
		XDestroyWindow(gpDisplay, window);
	} 
	
	if(colorMap) 
	{
		XFreeColormap(gpDisplay, colorMap);
	} 
	
	if(gpDisplay) 
	{
		XCloseDisplay(gpDisplay); 
		gpDisplay = NULL; 
	} 
}

