# GTA-Crime construction
## Requirement
- GTAVisionNative (compile from source [here](https://github.com/umautobots/GTAVisionExport/tree/master/native) or download x64 [here](https://github.com/umautobots/GTAVisionExport/files/1703454/native64bit.zip))
- ScriptHookV (download [here](http://www.dev-c.com/gtav/scripthookv/))
- ScriptHookVDotNet V3 (download [here](https://github.com/crosire/scripthookvdotnet/releases))
- Menyoo GTA5 trainer (download [here](https://www.gta5-mods.com/scripts/menyoo-pc-sp))
- Visual Studio
- others managed by Nuget

## Building C# code
- Git clone '''data_generation''' code
- Open .sin code in Visual Studio
- Right click Solution 'GTACrime' itself and go to: properties -> Configuration Properties and set the configuration to Release for x64
- Right click solution and click Build Solution
- The necessary files are created in './GTACrime/bin/x64/Release'.
 
## Preparation before running GTA5
- Install Grand Theft Auto V
- Copy 'ScriptHookV', 'ScriptHookVDotNet V3' files and Menyoo GTA5 trainer folder to 'Grand Theft Auto V' folder
- Make 'Scripts' folder in 'Grand Theft Auto V' folder
- Copy 'GTAVisionNative' files and necessary files created from construction code to 'Scripts' folder
- Set the 'gta_config.ini' file and copy to 'Scripts' folder
- In 'gta_config.ini', you can change various parameters such as save path, number of frames, occurrence events, and weather.

## Running GTA5
- Press 'PageUp' to activate plugin.
- F3 to enter the free camera mode.
- '0' to set random seed.
- F10 to save location. When you find a new location press 'L' to save it, a popup will ask for a name so you can find it again later. Make sure these names are unique. Using '[ ]' keys, you can cycle through already existing locations. Location files are saved in the path specified in the 'gta_config.ini' file.
- F11 to assign ROI. Use 'U' to update the ROI (note: ROI is defined as a closed polygon). You can use 'K' to remove the last corner point.
- Use 'N' to save all the locations. (required!!)
- F12 to collect data.

## Video creation
- After running GTA5, two viewpoint frames are created for each location in the path specified in 'gta_config.ini'.
- Videos can be created by synthesizing the frames using the provided Python code.

## Note
- An error may occur when updating GTA5.
- Most errors can be resolved by downloading the latest version of 'ScriptHookV' and 'ScriptHookVDotNet V3'.
