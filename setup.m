close all;
clear;
clc;

currentPath = pwd;
codePath = [currentPath '/code/'];
experimentsPath = [currentPath '/experiments/'];
dataPath = [currentPath '/data/'];

addpath(codePath);
addpath([codePath 'functions/']);
addpath([codePath 'functions/solvers/']);
addpath([codePath 'functions/rank_approximation/']);
addpath([codePath 'functions/initialization/']);
addpath([codePath 'functions/losses/']);
addpath(experimentsPath);
addpath(dataPath);