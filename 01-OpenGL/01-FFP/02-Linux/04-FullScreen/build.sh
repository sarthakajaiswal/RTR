rm *.o

gcc -c -o xwindow.o xwindow.c
gcc -o xwindow xwindow.o -lX11
./xwindow
