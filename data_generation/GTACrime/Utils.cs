using GTA;
using SharpDX;
using System;
using System.Collections.Generic;
using static GTACrime.EventExporter;

namespace GTACrime
{
    class Utils
    {
        private static Random random;

        public static GTA.Math.Vector3 RaycastFromCoord(float x, float y, Entity ignore, float maxDist, float failDist) // 0, 0 is center screen
        {
            return RaycastFromCoord(x, y, ignore, maxDist, failDist, out RaycastResult res);
        }

        public static GTA.Math.Vector3 RaycastFromCoord(float x, float y, Entity ignore, float maxDist, float failDist, out RaycastResult res) // 0, 0 is center screen
        {
            Camera camera = World.RenderingCamera;
            GTA.Math.Vector3 position = camera.Position;
            GTA.Math.Vector2 screenCoords = new GTA.Math.Vector2(x, y);

            GTA.Math.Vector3 WorldCoord = ScreenToWorld(screenCoords, camera);

            GTA.Math.Vector3 vector31 = WorldCoord - position;
            vector31.Normalize();

            res = World.Raycast(position + (vector31 * 1.0f), position + (vector31 * maxDist), (IntersectFlags)287, ignore);
            if (res.DidHit)
            {
                return res.HitPosition;
            }
            return position + (vector31 * failDist);
        }

        public static GTA.Math.Vector3 ScreenToWorld(GTA.Math.Vector2 screenCoords, Camera camera)
        {
            GTA.Math.Vector3 position = camera.Position;
            GTA.Math.Vector3 rotation = camera.Rotation;
            GTA.Math.Vector2 vector2;
            GTA.Math.Vector2 vector21;

            GTA.Math.Vector3 direction = camera.Direction;
            GTA.Math.Vector3 vector3 = rotation + new GTA.Math.Vector3(10.0f, 0.0f, 0.0f);
            GTA.Math.Vector3 vector31 = rotation + new GTA.Math.Vector3(-10.0f, 0.0f, 0.0f);
            GTA.Math.Vector3 vector32 = rotation + new GTA.Math.Vector3(0.0f, 0.0f, -10.0f);

            GTA.Math.Vector3 direction1 = RotationToDirection(rotation + new GTA.Math.Vector3(0.0f, 0.0f, 10.0f)) - RotationToDirection(vector32);
            GTA.Math.Vector3 direction2 = RotationToDirection(vector3) - RotationToDirection(vector31);

            float rad = -DegreeToRadian(rotation.Y);

            GTA.Math.Vector3 vector33 = GTA.Math.Vector3.Multiply(direction1, (float)Math.Cos(rad)) - GTA.Math.Vector3.Multiply(direction2, (float)Math.Sin(rad));
            GTA.Math.Vector3 vector34 = GTA.Math.Vector3.Multiply(direction1, (float)Math.Sin(rad)) + GTA.Math.Vector3.Multiply(direction2, (float)Math.Cos(rad));

            GTA.Math.Vector3 res1 = position + (direction * 10f);
            if (!WorldToScreenRel(res1, out vector21))
            {
                return res1;
            }
            if (!WorldToScreenRel(res1 + vector33 + vector34, out vector2))
            {
                return res1;
            }
            if (Math.Abs(vector2.X - vector21.X) < 0.001 || Math.Abs(vector2.Y - vector21.Y) < 0.001)
            {
                return res1;
            }
            float x = (screenCoords.X - vector21.X) / (vector2.X - vector21.X);
            float y = (screenCoords.Y - vector21.Y) / (vector2.Y - vector21.Y);
            return res1 + (vector33 * x) + (vector34 * y);
        }
        public static GTA.Math.Vector3 RotationToDirection(GTA.Math.Vector3 rot)
        {
            float retz = rot.Z * 0.0174532924F; // Degree to radian
            float retx = rot.X * 0.0174532924F;
            double absx = Math.Abs(Math.Cos(retx));
            return new GTA.Math.Vector3((float)(-Math.Sin(retz) * absx), (float)(Math.Cos(retz) * absx), (float)(Math.Sin(retx)));
        }
        public static float DegreeToRadian(float degrees)
        {
            return degrees * 0.0174532925199433F;
        }
        public static float RadianToDegree(float rads)
        {
            return rads / 0.0174532925199433F;
        }
        public static bool WorldToScreenRel(GTA.Math.Vector3 worldCoords, out GTA.Math.Vector2 screenCoords)
        {
            bool success;
            screenCoords = HashFunctions.Convert3dTo2d(worldCoords, out success);
            if (!success)
            {
                return false;
            }
            screenCoords.X = (screenCoords.X - 0.5f) * 2.0f;
            screenCoords.Y = (screenCoords.Y - 0.5f) * 2.0f;
            return success;
        }
        public static bool PointInPoly(GTALocation loc, GTAVector v)
        {
            int max_point = loc.ROI.Count - 1;
            float total_angle = GetAngle(loc.ROI[max_point].X, loc.ROI[max_point].Y, v.X, v.Y, loc.ROI[0].X, loc.ROI[0].Y);
            for (int i = 0; i < max_point; i++)
            {
                total_angle += GetAngle(
                    loc.ROI[i].X, loc.ROI[i].Y,
                    v.X, v.Y,
                    loc.ROI[i + 1].X, loc.ROI[i + 1].Y);
            }
            return Math.Abs(total_angle) > 1;
        }
        // Get angle between 3 points in 2D space
        public static float GetAngle(float Ax, float Ay, float Bx, float By, float Cx, float Cy)
        {
            float dot_product = DotProduct(Ax, Ay, Bx, By, Cx, Cy);
            float cross_product = CrossProductLength(Ax, Ay, Bx, By, Cx, Cy);
            return (float)Math.Atan2(cross_product, dot_product);
        }
        public static float CrossProductLength(float Ax, float Ay, float Bx, float By, float Cx, float Cy)
        {
            float BAx = Ax - Bx;
            float BAy = Ay - By;
            float BCx = Cx - Bx;
            float BCy = Cy - By;
            return BAx * BCy - BAy * BCx;
        }
        public static float DotProduct(float Ax, float Ay, float Bx, float By, float Cx, float Cy)
        {
            float BAx = Ax - Bx;
            float BAy = Ay - By;
            float BCx = Cx - Bx;
            float BCy = Cy - By;
            return BAx * BCx + BAy * BCy;
        }
        // Generate random point in polygon
        public static GTAVector RandPointInPoly(GTALocation x, int seed)
        {
            random = new Random(seed);  
            //random = new Random();

            float maxX = float.MinValue;
            float maxY = float.MinValue;
            float minX = float.MaxValue;
            float minY = float.MaxValue;
            float avgZ = 0f;
            foreach (GTAVector v in x.ROI)
            {
                maxX = Math.Max(maxX, v.X);
                maxY = Math.Max(maxY, v.Y);
                minX = Math.Min(minX, v.X);
                minY = Math.Min(minY, v.Y);
                avgZ += v.Z;
            }
            avgZ /= x.ROI.Count;
            avgZ += 0.5f;
            while (true)
            {
                GTAVector v = new GTAVector(random.NextFloat(minX, maxX), random.NextFloat(minY, maxY), avgZ);
                if (PointInPoly(x, v))
                    return v;
            }
        }
        public static double EuclideanDistance(GTAVector p1, GTAVector p2)
        {
            return Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
        }
        public static float PolygonArea(List<GTAVector> ROI)
        {
            int count = ROI.Count;
            GTAVector[] pts = new GTAVector[count + 1];
            ROI.CopyTo(pts, 0);
            pts[count] = ROI[0];

            double area = 0;
            for (int i = 0; i < count; i++)
            {
                area +=
                    (pts[i + 1].X - pts[i].X) *
                    (pts[i + 1].Y + pts[i].Y) / 2;
            }
            return (float)Math.Abs(area);
        }

        public static double[] ToDouble(float[] input)
        {
            return Array.ConvertAll(input, x => (double)x);
        }
    }
}
