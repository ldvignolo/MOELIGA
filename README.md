# README

## Multi-Objective Version of ELIGA - MOELIGA

- TODO
- Pass the k parameter of the tournament method from the configuration file (currently hardcoded)

To compile:

- Remember that MOELIGA must be compiled with the same compiler as MPICH (and mpd)[^1]

Execution:

- `>> mpdboot`
- `>> ./agp cfg archivo_SETTINGS.cfg`
- Or to persist after closing the session: `>> ./agp </dev/null &`[^1]

***

Compile MPICH in a path without spaces (**No need to compile as root, but yes for install**)

- `>> ./configure --with-pm=mpd`
- `>> make`
- `>> make install` (run as *root*)[^1]

**NOTE** If `sort` fails, install coreutils (FC25)

Then run:

- `>> cd src/pm/mpd`
- `>> ./configure`
- `>> make`
- `>> make install` (run as *root*)[^1]

Create file:

`>> vim /home/<user_name>/.mpd.conf` (set secretword=<...>)

`>> chmod 600 /home/<user_name>/.mpd.conf`

Then, to test (from anywhere):

- `>> mpdboot` (should have no output if working correctly)
- To check if the daemon starts: `>> mpdtrace` (should show hostname if working)
- `>> mpdallexit` (kills the process)[^1]

[^1]: README.md
