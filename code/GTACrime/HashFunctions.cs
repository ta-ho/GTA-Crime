using GTA;
using GTA.Math;
using GTA.Native;
using System.Collections.Generic;

namespace GTACrime
{
    public class HashFunctions
    {
        public static bool IsOnScreen(Vector3 pos)
        {
            return Function.Call<bool>(Hash.GET_SCREEN_COORD_FROM_WORLD_COORD, (InputArgument)pos.X, (InputArgument)pos.Y, (InputArgument)pos.Z, (InputArgument)new OutputArgument(), (InputArgument)new OutputArgument());
        }

        public static bool IsOnScreen(Entity e)
        {
            foreach (EntityBone bone in e.Bones)
            {
                try
                {
                    var pos = bone.Position;
                    bool success = Function.Call<bool>(Hash.GET_SCREEN_COORD_FROM_WORLD_COORD, (InputArgument)pos.X, (InputArgument)pos.Y, (InputArgument)pos.Z, (InputArgument)new OutputArgument(), (InputArgument)new OutputArgument());
                    if (success)
                    {
                        return true;
                    }
                }
                catch
                {
                    continue;
                }
            }
            return false;
        }

        public static Vector2 Convert3dTo2d(Vector3 pos, out bool success)
        {
            OutputArgument tmpResX = new OutputArgument();
            OutputArgument tmpResY = new OutputArgument();

            success = Function.Call<bool>(Hash.GET_SCREEN_COORD_FROM_WORLD_COORD, (InputArgument)pos.X, (InputArgument)pos.Y, (InputArgument)pos.Z, (InputArgument)tmpResX, (InputArgument)tmpResY);

            if (success)
            {
                Vector2 v2;
                v2.X = tmpResX.GetResult<float>();
                v2.Y = tmpResY.GetResult<float>();
                return v2;
            }
            else
                return new Vector2(-1f, -1f);
        }
        public static Vector2 Convert3dTo2d(GTAVector pos, out bool success)
        {
            OutputArgument tmpResX = new OutputArgument();
            OutputArgument tmpResY = new OutputArgument();

            success = Function.Call<bool>(Hash.GET_SCREEN_COORD_FROM_WORLD_COORD, (InputArgument)pos.X, (InputArgument)pos.Y, (InputArgument)pos.Z, (InputArgument)tmpResX, (InputArgument)tmpResY);

            if (success)
            {
                Vector2 v2;
                v2.X = tmpResX.GetResult<float>();
                v2.Y = tmpResY.GetResult<float>();
                return v2;
            }
            else
                return new Vector2(-1f, -1f);
        }
        public static void Draw3DLine(Vector3 iniPos, Vector3 finPos, byte col_r = 255, byte col_g = 255, byte col_b = 255, byte col_a = 255)
        {
            Function.Call(Hash.DRAW_LINE, new InputArgument[]
            {
                iniPos.X,
                iniPos.Y,
                iniPos.Z,
                finPos.X,
                finPos.Y,
                finPos.Z,
                (int)col_r,
                (int)col_g,
                (int)col_b,
                (int)col_a
            });
        }
        public static void Draw3DLine(GTAVector iniPos, GTAVector finPos, byte col_r = 255, byte col_g = 255, byte col_b = 255, byte col_a = 255)
        {
            Function.Call(Hash.DRAW_LINE, new InputArgument[]
            {
                iniPos.X,
                iniPos.Y,
                iniPos.Z,
                finPos.X,
                finPos.Y,
                finPos.Z,
                (int)col_r,
                (int)col_g,
                (int)col_b,
                (int)col_a
            });
        }
        public static void DrawRect(float x, float y, float w, float h, byte r = 255, byte g = 255, byte b = 255, byte a = 255)
        {
            Function.Call(Hash.DRAW_RECT, new InputArgument[] {
                x, y,
                w, h,
                (int)r, (int)g, (int)b, (int)a
            });
        }

        public static bool HAS_ENTITY_CLEAR_LOS_TO_ENTITY(Entity e1, Entity e2)
        {
            return Function.Call<bool>(Hash.HAS_ENTITY_CLEAR_LOS_TO_ENTITY, new InputArgument[] { e1, e2, 17 });
        }

        public static float getDotVectorResult(Ped source, Ped target)
        {
            if (source.Exists() && target.Exists())
            {
                Vector3 dir = (target.Position - source.Position).Normalized; return Vector3.Dot(dir, source.ForwardVector);
            }
            else
                return -1.0f;
        }

        public static bool IS_ANY_PED_IN_LOS(List<Ped> peds, float minAngle, bool withOcclusion = true, bool includeDead = false)
        {
            foreach (Ped ped in peds)
            {
                if (ped.Exists())
                {
                    if (ped.IsDead && !includeDead) { continue; }
                    if (withOcclusion) // with obstacle detection
                    {
                        if (HAS_ENTITY_CLEAR_LOS_TO_ENTITY(ped, Game.Player.Character)) // No Obstacles? 
                        {
                            float dot = getDotVectorResult(ped, Game.Player.Character);
                            if (dot > minAngle) // Is in acceptable range for dot product?
                            {
                                return true;
                            }
                        }
                    }
                    else // without obstacle detection 
                    {
                        float dot = getDotVectorResult(ped, Game.Player.Character);
                        if (dot > minAngle) // Is in acceptable range for dot product? 
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
    }
}

