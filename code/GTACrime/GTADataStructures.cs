using GTA;
using System.Diagnostics;
using System;
using System.Collections.Generic;
using System.Linq;
using Vector2 = GTA.Math.Vector2;
using Vector3 = GTA.Math.Vector3;
using MathNet.Numerics.LinearAlgebra.Double;

namespace GTACrime
{
    public class GTABoundingBox2
    {
        public GTAVector2 Min { get; set; }
        public GTAVector2 Max { get; set; }
    }

    public class GTABoundingBox
    {
        public GTAVector[] CornerPoints { get; }
        private readonly float x_max = 0, x_min = float.MaxValue, y_max = 0, y_min = float.MaxValue, z_max = 0, z_min = float.MaxValue;
        public GTABoundingBox(GTAVector[] points)
        {
            CornerPoints = points;
            foreach (GTAVector v in points)
            {
                x_max = Math.Max(x_max, v.X);
                x_min = Math.Min(x_min, v.X);

                y_max = Math.Max(y_max, v.Y);
                y_min = Math.Min(y_min, v.Y);

                z_max = Math.Max(z_max, v.Z);
                z_min = Math.Min(z_min, v.Z);
            }
        }

        public GTABoundingBox(Entity e)
        {
            var dim = new GTAVector();
            var e1 = new GTAVector();
            var e2 = new GTAVector();
            var e3 = new GTAVector();
            var e4 = new GTAVector();
            var e5 = new GTAVector();
            var e6 = new GTAVector();
            var e7 = new GTAVector();
            var e8 = new GTAVector();
            (var gmin, var gmax) = e.Model.Dimensions;

            var rightVector = e.ForwardVector;
            var upVector = e.UpVector;
            var forwardVector = e.RightVector;

            var position = e.Position;

            //Calculate siZe
            dim.X = 0.5f * (gmax.X - gmin.X);
            dim.Y = 0.5f * (gmax.Y - gmin.Y);
            dim.Z = 0.5f * (gmax.Z - gmin.Z);

            e1.X = position.X - dim.Y * rightVector.X - dim.X * forwardVector.X - dim.Z * upVector.X;
            e1.Y = position.Y - dim.Y * rightVector.Y - dim.X * forwardVector.Y - dim.Z * upVector.Y;
            e1.Z = position.Z - dim.Y * rightVector.Z - dim.X * forwardVector.Z - dim.Z * upVector.Z;

            e2.X = e1.X + 2 * dim.Y * rightVector.X;
            e2.Y = e1.Y + 2 * dim.Y * rightVector.Y;
            e2.Z = e1.Z + 2 * dim.Y * rightVector.Z;

            e3.X = e2.X + 2 * dim.Z * upVector.X;
            e3.Y = e2.Y + 2 * dim.Z * upVector.Y;
            e3.Z = e2.Z + 2 * dim.Z * upVector.Z;

            e4.X = e1.X + 2 * dim.Z * upVector.X;
            e4.Y = e1.Y + 2 * dim.Z * upVector.Y;
            e4.Z = e1.Z + 2 * dim.Z * upVector.Z;

            e5.X = position.X + dim.Y * rightVector.X + dim.X * forwardVector.X + dim.Z * upVector.X;
            e5.Y = position.Y + dim.Y * rightVector.Y + dim.X * forwardVector.Y + dim.Z * upVector.Y;
            e5.Z = position.Z + dim.Y * rightVector.Z + dim.X * forwardVector.Z + dim.Z * upVector.Z;

            e6.X = e5.X - 2 * dim.Y * rightVector.X;
            e6.Y = e5.Y - 2 * dim.Y * rightVector.Y;
            e6.Z = e5.Z - 2 * dim.Y * rightVector.Z;

            e7.X = e6.X - 2 * dim.Z * upVector.X;
            e7.Y = e6.Y - 2 * dim.Z * upVector.Y;
            e7.Z = e6.Z - 2 * dim.Z * upVector.Z;

            e8.X = e5.X - 2 * dim.Z * upVector.X;
            e8.Y = e5.Y - 2 * dim.Z * upVector.Y;
            e8.Z = e5.Z - 2 * dim.Z * upVector.Z;

            CornerPoints = new GTAVector[] { e1, e2, e3, e4, e5, e6, e7, e8 };

            foreach (GTAVector v in CornerPoints)
            {
                x_max = Math.Max(x_max, v.X);
                x_min = Math.Min(x_min, v.X);

                y_max = Math.Max(y_max, v.Y);
                y_min = Math.Min(y_min, v.Y);

                z_max = Math.Max(z_max, v.Z);
                z_min = Math.Min(z_min, v.Z);
            }
        }

        public bool Contains(GTAVector p)
        {
            return (p.X <= x_max && p.X >= x_min) && (p.Y <= y_max && p.Y >= y_min) && (p.Z <= z_max && p.Z >= z_min);
        }

        public bool Contains(Vector3 p)
        {
            return (p.X <= x_max && p.X >= x_min) && (p.Y <= y_max && p.Y >= y_min) && (p.Z <= z_max && p.Z >= z_min);
        }

        public GTABoundingBox2 To2D()
        {
            float x_min = float.MaxValue, x_max = 0, y_min = float.MaxValue, y_max = 0;
            foreach (GTAVector v in CornerPoints)
            {
                Vector2 temp = HashFunctions.Convert3dTo2d(v, out bool success);
                if (success)
                {
                    x_max = Math.Max(x_max, temp.X);
                    x_min = Math.Min(x_min, temp.X);

                    y_max = Math.Max(y_max, temp.Y);
                    y_min = Math.Min(y_min, temp.Y);
                }
                else
                {
                    //System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", "Coord failed: " + temp.ToString() + "\n");
                }
            }

            x_max = Math.Min(x_max, 1f);
            x_min = Math.Max(x_min, 0f);

            y_max = Math.Min(y_max, 1f);
            y_min = Math.Max(y_min, 0f);

            var result = new GTABoundingBox2
            {
                Min = new GTAVector2(x_min, y_min),
                Max = new GTAVector2(x_max, y_max)
            };
            return result;
        }
    }

    public enum DetectionType
    {
        background,
        person,
        car,
        //    bike,
        //    bicycle,
        //    bus,
        //    van,
        //    train,
        entity
    }

    public class GTADetection
    {
        public DetectionType Type { get; set; }
        public float Distance { get; set; }
        public int Handle { get; set; }
        public int SeatIndex { get; set; }
        public int Gender { get; set; }
        public int VehicleClass { get; set; }
        public GTABoundingBox2 Bbox2d { get; set; }
        public GTABoundingBox Bbox3d { get; set; }
        //public Dictionary<string, GTAVector2> Bones2D { get; set; }
        public Dictionary<string, GTABone> Bones3D { get; set; }
        public GTADetection(Entity e, DetectionType type, float maxRange)
        {
            Type = type;
            var ppos = World.RenderingCamera.Position;

            Distance = ppos.DistanceTo(e.Position);
            Handle = e.Handle;
            //Bones2D = new Dictionary<string, GTAVector2>();
            Bones3D = new Dictionary<string, GTABone>();
            //(var gmin, var gmax) = e.Model.Dimensions;
            Bbox3d = new GTABoundingBox(e);
            Bbox2d = Bbox3d.To2D();
            //System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", "------------" + Handle.ToString() + "------------\n");
            RaycastResult res;
            Entity hitEntity;
            GTABone gtaBone;
            foreach (EntityBone bone in e.Bones)
            {
                try
                {
                    Vector3 bonePosition = bone.Position;
                    float boneDistance = ppos.DistanceTo(bonePosition);
                    
                    // Raycasting seems to be bugged where some people have no collision? Disabled this for now...

                    //if (boneDistance < maxRange)
                    //{
                    //    res = World.Raycast(ppos, bonePosition, (IntersectFlags)287, Game.Player.Character);
                    //    hitEntity = res.HitEntity;

                    //    int isVisible = 0;
                    //    if (hitEntity == null)
                    //    {
                    //        isVisible = -1;
                    //    }
                    //    else if (hitEntity.Handle == e.Handle)
                    //    {
                    //        isVisible = 1;
                    //    }
                    //    gtaBone = new GTABone(bone, isVisible, res);
                    //}
                    //else
                    //{
                    //    gtaBone = new GTABone(bone, -2, false, null);
                    //}

                    gtaBone = new GTABone(bone);
                    Bones3D.Add(bone.Index.ToString(), gtaBone);
                }
                catch
                {
                    continue;
                }
            }
        }

        public GTADetection(Ped p, float maxRange) : this(p, DetectionType.person, maxRange) 
        {
            this.SeatIndex = (int) p.SeatIndex;
            this.Gender = (int)p.Gender;
        }
        public GTADetection(Vehicle v, float maxRange) : this(v, DetectionType.car, maxRange)
        {
            this.VehicleClass = (int) v.ClassType;
        }
        public static GTADetection CreateDetection(Entity e, float maxRange=120f)
        {
            if (e is Ped p) 
                return new GTADetection(p, DetectionType.person, maxRange);
            else if (e is Vehicle v)
                return new GTADetection(v, DetectionType.car, maxRange);
            else 
                return new GTADetection(e, DetectionType.entity, maxRange);
        }
    }
   
    public class GTAData
    {
        public List<GTADetection> Detections { get; set; }

        private static IEnumerable<Entity> GetNearbyEntities(float radius = 200f)
        {
            var entities = World.GetNearbyEntities(World.RenderingCamera.Position, radius);
            return from entity in entities
                             where entity != null && (entity.Model.IsBicycle || entity.Model.IsBike || entity.Model.IsVehicle || entity.Model.IsPed) && entity.Handle != Game.Player.Character.Handle && HashFunctions.IsOnScreen(entity)
                             select entity;
        }

        public static GTAData DumpPedsData()
        {
            var ret = new GTAData();
            Ped[] peds = World.GetNearbyPeds(World.RenderingCamera.Position, 500f);
            var pedList = from ped in peds
                          where ped != null && ped.IsHuman && ped.IsVisible && !ped.IsOccluded
                          select new GTADetection(ped, 120f);
            ret.Detections = new List<GTADetection>();
            foreach (GTADetection d in pedList.ToList())
            {
                if (d.Bones3D.Count > 3)
                    ret.Detections.Add(d);
            }
            return ret;
        }

        public static void ShowBoundingBoxes(List<Ped> spawned=null)
        {
            
            List<Entity> entityList;
            if (spawned == null) 
            {
                entityList = GetNearbyEntities(200f).ToList();
            }
            else
            {
                entityList = new List<Entity>();
                foreach (Ped p in spawned)
                {
                    entityList.Add((Entity)p);
                }
            }

            var ppos = World.RenderingCamera.Position;
            var LinePos = new Vector3(ppos.X, ppos.Y, ppos.Z - 0.5f) + World.RenderingCamera.ForwardVector.Normalized;

            //int cnt = 0;
            //int failed = 0;
            float maxRange = 200f;

            RaycastResult res;
            Entity hitEntity;
            System.Drawing.Color color;
            
            foreach (Entity e in entityList)
            {
                if (!HashFunctions.IsOnScreen(e))
                    continue;

                Vector3 ePos = e.Position;
                float eDistance = ppos.DistanceTo(ePos);
                if (eDistance < maxRange)
                {
                    int cnt = 0;
                    foreach (EntityBone bone in e.Bones)
                    {
                        try
                        {
                            if (cnt++ % 2 == 0)
                                continue;
                            res = World.Raycast(LinePos, bone.Position, IntersectFlags.Everything, Game.Player.Character);
                            hitEntity = res.HitEntity;
                            color = System.Drawing.Color.Red;
                            //if (hitEntity == null)
                            //{
                            //    color = System.Drawing.Color.Yellow;
                            //}
                            //else if (hitEntity.Handle == e.Handle)
                            //{
                            //    color = System.Drawing.Color.Green;
                            //}
                            World.DrawLine(LinePos, bone.Position, color);
                        }
                        catch
                        {
                            continue;
                        }
                    }
                }
                else
                {
                    color = System.Drawing.Color.Blue;
                    World.DrawLine(LinePos, e.Position, color);
                }
            }
                
                //res = World.Raycast(LinePos, bonePosition, (IntersectFlags)287, Game.Player.Character);
                //hitEntity = res.HitEntity;
                //System.Drawing.Color color = System.Drawing.Color.Red;
                //if (hitEntity == null)
                //{
                //    color = System.Drawing.Color.Yellow;
                //}
                //else if (hitEntity.Handle == e.Handle)
                //{
                //    color = System.Drawing.Color.Green;
                //}
                //World.DrawLine(LinePos, bone.Position, color);
                //var gtaBone = new GTABone(bone, isVisible, res);

            //foreach (Entity e in entityList)
            //{
            //    var stopwatch = new Stopwatch();
            //    stopwatch.Start();
                //var boundingBox = new GTABoundingBox(e);
                //var corners = boundingBox.CornerPoints;
                //for (int i = 0; i < corners.Length; ++i)
                //{
                //    for (int j = 0; j < corners.Length; ++j)
                //    {
                //        if (j == i) continue;
                //        var c1 = corners[i];
                //        var c2 = corners[j];
                //        HashFunctions.Draw3DLine(c1, c2, 0, 0, 255);
                //    }
                //}
                //var isLOS = Function.Call<bool>((Hash)0x0267D00AF114F17A, Game.Player.Character, e);
                //World.DrawLine(e.Position, e.Position + e.ForwardVector.Normalized, System.Drawing.Color.Blue);
                //List<int> visibleBones = new List<int>();

                //if (e is Ped p)
                //{
                //    foreach (string name in BoneNames)
                //    {
                //        Bone temp = (Bone)Enum.Parse(typeof(Bone), name, true);
                //        var bone = p.Bones[temp];
                //        if (!bone.IsValid)
                //            continue;
                //        var res = World.Raycast(ppos, bone.Position, (IntersectFlags)287, Game.Player.Character);
                //        Entity hitEntity = res.HitEntity;
                        
                //        if (hitEntity.Handle == p.Handle)
                //        {
                //            int successCounter = 0;
                //            while (true)
                //            {
                //                res = World.Raycast(ppos, bone.Position, (IntersectFlags)287, Game.Player.Character);
                //                hitEntity = res.HitEntity;
                //                if (hitEntity != null && hitEntity.Handle == p.Handle)
                //                {
                //                    successCounter++;
                //                    continue;
                //                }
                //                break;
                //            }
                //            System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", e.EntityType.ToString() + " : failed after " + successCounter + "raycasts.\n");
                //            return;
                //        }

                        //int cnt = 0;
                        //while (hitEntity == null && cnt < 10)
                        //{
                        //    Script.Wait(10);
                        //    res = World.Raycast(ppos, bone.Position, (IntersectFlags)287, Game.Player.Character);
                        //    hitEntity = res.HitEntity;
                        //    cnt += 1;
                        //}

                        //if (cnt == 10)
                        //{
                        //    System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", e.EntityType.ToString() + " : hit entity still null after 10 casts.\n");
                        //    continue;                        
                        //}

                //        bool visible = false;

                //        if (hitEntity.Handle == p.Handle)
                //        {
                //            visible = true;
                //            //visibleBones.Add(bone.Index);
                //        }
                //    }
                //    if (failed == 0)
                //    {
                //        HashFunctions.Draw3DLine(linePos, e.Position, 0, 0, 255);
                //    }
                //}
                //else
                //{
                //    continue;
                //}
                //else if (e is Vehicle v)
                //{
                //    foreach (EntityBone bone in v.Bones)
                //    {
                //        var res = World.Raycast(ppos, bone.Position, (IntersectFlags)287, Game.Player.Character);
                //        var hitEntity = res.HitEntity;
                //        bool visible = false;

                //        if (hitEntity != null && hitEntity.Handle == e.Handle)
                //        {
                //            visible = true;
                //            //visibleBones.Add(bone.Index);
                //        }
                //    }
                //}
                //stopwatch.Stop();
                //var elapsed = stopwatch.ElapsedMilliseconds;
                //var count = e.Bones.Count;
                //var visbleCount = visibleBones.Count;
                //System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", e.EntityType.ToString() + " doesnt have " + string.Join(",", missingBones) + "\n"); ;
                //System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", Bone.BagBody.ToString() + "\n"); ;
                //else
                //World.DrawMarker(MarkerType.DebugSphere, e.Position, Vector3.Zero, Vector3.Zero, new Vector3(0.5f, 0.5f, 0.5f), System.Drawing.Color.Red);

                //foreach (EntityBone bone in e.Bones)
                //{

                //    if (boundingBox.Contains(bone.Position))
                //        {
                //            //World.DrawMarker(MarkerType.DebugSphere, bone.Position, Vector3.Zero, Vector3.Zero, new Vector3(0.05f, 0.05f, 0.05f), System.Drawing.Color.Green);
                //        }
                //        else
                //        {
                //            World.DrawMarker(MarkerType.DebugSphere, bone.Position, Vector3.Zero, Vector3.Zero, new Vector3(0.05f, 0.05f, 0.05f), System.Drawing.Color.Red);
                //        }
                //}
            //}
            //System.IO.File.AppendAllText("F:/datasets/GTA_V_anomaly/log.txt", cnt + " " + failed + ".\n");

        }

        public static GTAData DumpEntityData()
        {
            var entityList = GetNearbyEntities(500f);
            var ret = new GTAData
            {
                Detections = new List<GTADetection>()
            };

            foreach (Entity e in entityList)
            {
                var d = GTADetection.CreateDetection(e);
                if (d.Bones3D.Count > 3)
                {
                    ret.Detections.Add(d);
                }
            }
            return ret;
        }
    }
    public class GTABone
    {
        public GTAVector Pos { get; set; }
        public int IsVisible { get; set; }
        public bool DidHit { get; set; }
        public string Material { get; set; }
        public GTABone(EntityBone bone)
        {
            this.Pos = new GTAVector(bone.Position);
        }
        public GTABone(EntityBone bone, int isVisible, bool didHit, string material)
        {
            this.Pos = new GTAVector(bone.Position);
            this.IsVisible = isVisible;
            this.DidHit = didHit;
            this.Material = material;
        }
        public GTABone(EntityBone bone, int isVisible, RaycastResult res)
        {
            this.Pos = new GTAVector(bone.Position);
            this.IsVisible = isVisible;
            this.DidHit = res.DidHit;
            this.Material = res.MaterialHash.ToString();
        }
    }

    public class GTAVector
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public GTAVector()
        {
            X = 0;
            Y = 0;
            Z = 0;
        }
        public GTAVector(Vector3 v)
        {
            X = v.X;
            Y = v.Y;
            Z = v.Z;
        }

        public GTAVector(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static explicit operator SharpDX.Vector3(GTAVector i)
        {
            return new SharpDX.Vector3(i.X, i.Y, i.Z);
        }

        public static GTAVector Lerp(GTAVector from, GTAVector to, float by)
        {
            var x = from.X * (1 - by) + to.X * by;
            var y = from.Y * (1 - by) + to.Y * by;
            var z = from.Z * (1 - by) + to.Z * by;
            return new GTAVector(x, y, z);
        }
    }

    public class GTAMatrix
    {
        public double[] Values { get; set; }
        public int ColumnCount { get; set; }
        public int RowCount { get; set; }
        public GTAMatrix() { }
        public GTAMatrix(double[] values, int cols, int rows)
        {
            this.Values = values;
            this.ColumnCount = cols;
            this.RowCount = rows;
        }

        public GTAMatrix(DenseMatrix m)
        {
            this.Values = m.Values;
            this.RowCount = m.RowCount;
            this.ColumnCount = m.ColumnCount;
        }

        public static explicit operator DenseMatrix(GTAMatrix m)
        {
            return new DenseMatrix(m.RowCount, m.ColumnCount, m.Values);
        }
    }

    public class GTAVector2
    {
        public float X { get; set; }
        public float Y { get; set; }

        public GTAVector2(float x, float y)
        {
            X = x;
            Y = y;
        }
        public GTAVector2()
        {
            X = 0f;
            Y = 0f;
        }
        public GTAVector2(Vector2 v)
        {
            X = v.X;
            Y = v.Y;
        }
    }
}