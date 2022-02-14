# casim
Computer Architecture Simulation Infrastructure for CSCE 614 Computer Architecture



### 1. Environemnt setup

Everytime you want to build or run zsim, you need to setup the environment variables first.

```
$ source setup_env
```

##### 2. Compile zsim

```
$ cd zsim
$ scons -j4
```

##### 3 Launch a test to run

```
./build/opt/zsim tests/simple.cfg
```



###### For more information, check `zsim/README.md`
