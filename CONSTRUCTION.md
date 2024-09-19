# GTA-Crime construction
## Prerequisite file
- GTAVision (compile from source [here](https://github.com/umautobots/GTAVisionExport/tree/master/native) or download x64 [here](https://github.com/umautobots/GTAVisionExport/files/1703454/native64bit.zip))
- ScriptHookV (download [here](http://www.dev-c.com/gtav/scripthookv/))
- ScriptHookVDotNet V3 (download [here](https://github.com/crosire/scripthookvdotnet/releases))
- Menyoo GTA5 trainer (download [here](https://www.gta5-mods.com/scripts/menyoo-pc-sp))
- Visual Studio
- others managed by Nuget

  ## Build C# code
  - Git clone construction code
  - Open .sin code in Visual Studio
  - Right click Solution 'GTACrime' itself and go to: properties -> Configuration Properties and set the configuration to Release for x64
  - Right click solution and click Build Solution
  - The necessary files are created in './GTACrime/bin/x64/Release'.
 
  ## Preparation before launching GTA5
  - install Grand Theft Auto V
  - copy 'ScriptHookV', 'ScriptHookVDotNet V3' files and Menyoo GTA5 trainer folder to 'Grand Theft Auto V' folder
  - make 'Scripts' folder in 'Grand Theft Auto V' folder
  - copy 'GTAVisionNative' files and necessary files created from construction code to 'Scripts' folder
  - set 'gta_config.ini' file and copy to 'Scripts' folder
