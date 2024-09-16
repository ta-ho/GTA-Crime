using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace GTACrime
{
    class GrabScreen
    {
        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")]
        public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, int nFlags);
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            private int _Left;
            private int _Top;
            private int _Right;
            private int _Bottom;

            public RECT(RECT Rectangle) : this(Rectangle.Left, Rectangle.Top, Rectangle.Right, Rectangle.Bottom)
            {
            }
            public RECT(int Left, int Top, int Right, int Bottom)
            {
                _Left = Left;
                _Top = Top;
                _Right = Right;
                _Bottom = Bottom;
            }

            public int X
            {
                get { return _Left; }
                set { _Left = value; }
            }
            public int Y
            {
                get { return _Top; }
                set { _Top = value; }
            }
            public int Left
            {
                get { return _Left; }
                set { _Left = value; }
            }
            public int Top
            {
                get { return _Top; }
                set { _Top = value; }
            }
            public int Right
            {
                get { return _Right; }
                set { _Right = value; }
            }
            public int Bottom
            {
                get { return _Bottom; }
                set { _Bottom = value; }
            }
            public int Height
            {
                get { return _Bottom - _Top; }
                set { _Bottom = value + _Top; }
            }
            public int Width
            {
                get { return _Right - _Left; }
                set { _Right = value + _Left; }
            }
            public Point Location
            {
                get { return new Point(Left, Top); }
                set
                {
                    _Left = value.X;
                    _Top = value.Y;
                }
            }
            public Size Size
            {
                get { return new Size(Width, Height); }
                set
                {
                    _Right = value.Width + _Left;
                    _Bottom = value.Height + _Top;
                }
            }

            public static implicit operator Rectangle(RECT Rectangle)
            {
                return new Rectangle(Rectangle.Left, Rectangle.Top, Rectangle.Width, Rectangle.Height);
            }
            public static implicit operator RECT(Rectangle Rectangle)
            {
                return new RECT(Rectangle.Left, Rectangle.Top, Rectangle.Right, Rectangle.Bottom);
            }
            public static bool operator ==(RECT Rectangle1, RECT Rectangle2)
            {
                return Rectangle1.Equals(Rectangle2);
            }
            public static bool operator !=(RECT Rectangle1, RECT Rectangle2)
            {
                return !Rectangle1.Equals(Rectangle2);
            }

            public override string ToString()
            {
                return "{Left: " + _Left + "; " + "Top: " + _Top + "; Right: " + _Right + "; Bottom: " + _Bottom + "}";
            }

            public override int GetHashCode()
            {
                return ToString().GetHashCode();
            }

            public bool Equals(RECT Rectangle)
            {
                return Rectangle.Left == _Left && Rectangle.Top == _Top && Rectangle.Right == _Right && Rectangle.Bottom == _Bottom;
            }

            public override bool Equals(object Object)
            {
                if (Object is RECT)
                {
                    return Equals((RECT)Object);
                }
                else if (Object is Rectangle)
                {
                    return Equals(new RECT((Rectangle)Object));
                }

                return false;
            }
        }

        // Crop out windows non-sense; the windows bar on top and some 3 pixels they add around the edges?
        private static Rectangle section;
        //private static Bitmap resized_bitmap;
        private static Bitmap window_bitmap;
        private static bool section_null = true;


        public static Bitmap ScreenShot(string procName, int width, int height)
        {
            Process proc;
            try
            {
                proc = Process.GetProcessesByName(procName)[0];
                // Focus window so the screenshot is of the actual game content in case it gets covered
                SetForegroundWindow(proc.MainWindowHandle);
            }
            catch
            {
                // should never happen but still
                return null;
            }

            GetWindowRect(proc.MainWindowHandle, out RECT rc);

            // This window is too big (not 2560x1440) and we fix this later
            //Bitmap bmp = new Bitmap(rc.Width, rc.Height);
            if (section_null)
            {
                //File.AppendAllText("F:\\datasets\\GTA_V_anomaly\\log.txt", "screengrab: " + rc.Width + "x" + rc.Height + "\n");

                window_bitmap = new Bitmap(rc.Width, rc.Height);
                if (rc.Width == width && rc.Height == height)
                    section = new Rectangle(new Point(0, 0), new Size(rc.Width, rc.Height));
                else
                    section = new Rectangle(new Point(3, 32), new Size(rc.Width - 6, rc.Height - 35));
                section_null = false;
            }

            Bitmap resized_bitmap = new Bitmap(section.Width, section.Height);

            using (var g = Graphics.FromImage(window_bitmap))
            {
                g.CopyFromScreen(new Point(rc.Left, rc.Top), Point.Empty, rc.Size);
            }

            using (var g = Graphics.FromImage(resized_bitmap))
            {
                g.DrawImage(window_bitmap, 0, 0, section, GraphicsUnit.Pixel);
            }
            // Final screenshot (2560x1440 in my case)
            return resized_bitmap;
        }
    }
}
