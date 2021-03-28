#!/usr/bin/env python3
import math
import random
#import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
import sys
#import tsp_utils
#import animated_visualizer

#S = 200


class SimulatedAnnealing:
    def __init__(self, distmatrix, alpha,stopping_iter,listlambadesired):
        ''' animate the solution over time

            Parameters
            ----------
            coords: array_like
                list of coordinates
            temp: float
                initial temperature
            alpha: float
                rate at which temp decreases
            stopping_temp: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates

        '''
        self.timer = []
        self.candidate1 = 0
        self.candidate2 = 0
        self.candidate3 = 0
        self.candidate4 = 0
        self.candidate5 = 0
        self.candidate6 = 0
        self.candidate7 = 0
        self.sort = 0
        self.min = 0
        self.timeweight = 0
        self.timeweightdiffinverse = 0
        self.timeweightdiffswap = 0
        self.timeweightdiffdisplacement = 0
        self.timeweightdiffinsert = 0
        self.timeiterations = [0]
        self.start= time.time()
        self.sample_size = distmatrix.shape[0]
        self.oldsample_size = None
        ##print("sample:",self.sample_size)


        self.alpha = alpha
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = distmatrix
        #self.curr_solution = tsp_utils.nearestNeighbourSolution(self.dist_matrix)
        self.curr_solution =[0]+list(np.random.permutation(np.arange(1, self.sample_size)))
        self.best_solution = self.curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.old_weight = self.curr_weight
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]
        self.min_weightlist = [self.min_weight]
        self.initialtemp = 0
        self.templist = []
        self.initialtemplist = []
        self.temp = 1000
        self.counter = 0
        self.counter2 = 0
        self.counter3 = 0
        self.k = 0
        self.probability = 1
        self.listlambadesired = listlambadesired
        self.diffsample = distmatrix.shape[0]


    #
    # def weight(self, sol):
    #     '''
    #     Calcuate weight
    #     '''
    #     t0 = time.time()
    #     weight = sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])
    #     t1 = time.time()
    #     self.timeweight += t1-t0
    #     return weight
    #
    # def weight(self, individual):
    #     #     fitness = 0
    #     #     for i in range(len(individual)-1):
    #     #         fitness += self.dist_matrix[individual[i], individual[i+1]]
    #     #     return fitness
    def weight(self, individual):
        fitness = 0
        for i in range(len(individual)):
            ###print(individual)
            ###print(len(individual))
            fitness += self.dist_matrix[individual[i-1], individual[i]]
        return fitness

    def weightdiffinverse(self, current, candidate, i, j):
        if j == len(candidate) - 1:
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[
                candidate[j], candidate[0]]
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[j], current[0]]
        else:
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[j], current[j + 1]]
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[
                candidate[j], candidate[j + 1]]
        diff = weightafter - weightbefore
        return diff

    # def weightdiffinverse(self,current,candidate,i,j):
    #     t0 = time.time()
    #     weightbefore = 0
    #     weightafter = 0
    #     if j < self.sample_size-1:
    #         for k in range(i,j+2):
    #             weightbefore += self.dist_matrix[current[k-1],current[k]]
    #             weightafter += self.dist_matrix[candidate[k-1], candidate[k]]
    #             diff = weightafter - weightbefore
    #     else:
    #         if j == self.sample_size-1:
    #             for k in range(i,j+1):
    #                 weightbefore += self.dist_matrix[current[k-1],current[k]]
    #                 weightafter += self.dist_matrix[candidate[k-1], candidate[k]]
    #                 diff = (weightafter + self.dist_matrix[candidate[self.sample_size-1],candidate[0]]) - (weightbefore + self.dist_matrix[current[self.sample_size-1],current[0]])
    #     t1 = time.time()
    #     self.timeweightdiffinverse += t1 - t0
    #     # if diff - (self.weight(candidate) - self.weight(current)) < 10:
    #     #     print('correct weightdiffinverse')
    #     # else:
    #     #     print('not weightdiffinverse', diff - (self.weight(candidate) - self.weight(current)))
    #     #     print(current)
    #     #     print(candidate)
    #     #     print(i, current[i])
    #     #     print(j, current[j])
    #     return diff
    def weightdiffDisplacement(self,current,candidate,i,j,r):
        t0 = time.time()
        diff = 0

        if r == i:
            return 0

        if j < self.sample_size - 1 and r < self.sample_size - (j - i) - 1:
            if r < i:
                weightafter = self.dist_matrix[candidate[r - 1], candidate[r]] + self.dist_matrix[candidate[r + (j - i)], candidate[r + (j - i) + 1]] + \
                              self.dist_matrix[candidate[j], candidate[j + 1]]
            else:
                weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[r - 1], candidate[r]] \
                              + self.dist_matrix[candidate[r + (j - i)], candidate[r + (j - i) + 1]]
            if r > i:
                r = r + (j - i) + 1
            weightbefore = self.dist_matrix[current[r - 1], current[r]] + self.dist_matrix[
                current[i - 1], current[i]] + \
                           self.dist_matrix[current[j], current[j + 1]]
            diff = weightafter - weightbefore
            return diff
        elif r == self.sample_size - (j - i) - 1:
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + \
                           self.dist_matrix[current[j], current[j + 1]] + self.dist_matrix[
                               current[self.sample_size - 1], current[0]]
            weightafter = self.dist_matrix[
                              candidate[i + self.sample_size - 2 - j], candidate[i + self.sample_size - j - 1]] + \
                          self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[
                              candidate[self.sample_size - 1], candidate[0]]
            diff = weightafter - weightbefore
            return diff
        elif j == self.sample_size - 1 and r < i:
            weightbefore = self.dist_matrix[current[r - 1], current[r]] + self.dist_matrix[
                current[i - 1], current[i]] \
                           + self.dist_matrix[current[j], current[0]]
            weightafter = self.dist_matrix[candidate[r - 1], candidate[r]] + self.dist_matrix[
                candidate[r + (j - i)], candidate[r + (j - i) + 1]] \
                          + self.dist_matrix[candidate[j], candidate[0]]
            diff = weightafter - weightbefore
            return diff
        return diff;
        t1 = time.time()
        self.timeweightdiffdisplacement += t1 - t0
        return diff;
    def weightdiffswap(self,current,candidate,i,j):
        t0 = time.time()
        if i == j:
            return 0;
        if np.abs(j-i) == 1:
            if i < self.sample_size -1 and j < self.sample_size -1:
                s = min(i, j)
                weightbefore = self.dist_matrix[current[s - 1], current[s]] + self.dist_matrix[current[s], current[s + 1]] + \
                            + self.dist_matrix[current[s+1], current[s + 2]]
                weightafter = self.dist_matrix[candidate[s - 1], candidate[s]] + self.dist_matrix[candidate[s], candidate[s + 1]]  \
                              + self.dist_matrix[candidate[s+1], candidate[s + 2]]
                diff = weightafter - weightbefore
            else:
                s = min(i,j)
                weightbefore = self.dist_matrix[current[s - 1], current[s]] + self.dist_matrix[current[s], current[s + 1]] + \
                               + self.dist_matrix[current[s+1], current[0]]
                weightafter = self.dist_matrix[candidate[s - 1], candidate[s]] + self.dist_matrix[candidate[s], candidate[s + 1]] + \
                               + self.dist_matrix[candidate[s+1], candidate[0]]
                diff = weightafter - weightbefore
        if i < self.sample_size -1 and j < self.sample_size -1 and np.abs(j-i) != 1:
            weightbefore= self.dist_matrix[current[i-1], current[i]] + self.dist_matrix[current[i], current[i+1]] + \
                          self.dist_matrix[current[j-1], current[j]] + self.dist_matrix[current[j], current[j+1]]
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i], candidate[i + 1]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]] + self.dist_matrix[candidate[j], candidate[j + 1]]
            diff = weightafter - weightbefore
        elif i == self.sample_size-1 and np.abs(j-i) != 1:
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[self.sample_size-1], current[0]] + \
                           self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[current[j], current[j + 1]]
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[self.sample_size-1], candidate[0]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]] + self.dist_matrix[candidate[j], candidate[j + 1]]
            diff = weightafter - weightbefore
        elif j == self.sample_size-1 and np.abs(j-i) != 1:
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[i], current[i + 1]] + \
                           self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[current[self.sample_size-1], current[0]]
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i], candidate[i + 1]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]] + self.dist_matrix[candidate[self.sample_size-1], candidate[0]]
            diff = weightafter - weightbefore

        t1 = time.time()
        self.timeweightdiffswap += t1 - t0
        return diff

    def weightdiffinsert(self, current, candidate, i, j):
        if (i < j):
            if j < self.sample_size - 1:
                weightbefore = self.dist_matrix[current[i - 1], current[i]] + \
                               self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[current[j], current[j + 1]]
                weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i], candidate[i + 1]] + \
                              self.dist_matrix[candidate[j], candidate[j + 1]]
                diff = weightafter - weightbefore
                return diff

            else:
                weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[j - 1], current[j]] + \
                               self.dist_matrix[current[j], current[0]]
                weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i], candidate[i + 1]] + \
                              self.dist_matrix[candidate[j], candidate[0]]
                diff = weightafter - weightbefore
                return diff
        else:
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + \
                           self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[
                               current[j], current[j + 1]]
            weightafter = self.dist_matrix[candidate[i - 2], candidate[i - 1]] + self.dist_matrix[
                candidate[i - 1], candidate[i]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]]
            diff = weightafter - weightbefore
            return diff

    def weightdiffBigswap(self,current,candidate,i,j):
        diff = 0
        if i == j:
            return diff
        if abs(j-i) == 2:
            if i > j:
                i , j = j , i
            if j + 1 == self.sample_size - 1:
                weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[i + 1], current[i + 2]] + \
                               self.dist_matrix[current[j + 1], current[0]]
                weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i + 1], candidate[i + 2]] + \
                              self.dist_matrix[candidate[j + 1], candidate[0]]
            else:
                weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[i + 1], current[i + 2]] + \
                               self.dist_matrix[current[j + 1], current[j + 2]]
                weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i + 1], candidate[i + 2]] + \
                              self.dist_matrix[candidate[j + 1], candidate[j + 2]]
            diff = weightafter - weightbefore
            return diff
        if j + 1 == self.sample_size - 1 or i + 1 == self.sample_size - 1:
            if i > j:
                i , j = j , i
            weightbefore = self.dist_matrix[current[i - 1], current[i]] + self.dist_matrix[current[i + 1], current[i + 2]] + \
                           self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[current[j + 1], current[0]]
            weightafter = self.dist_matrix[candidate[i - 1], candidate[i]] + self.dist_matrix[candidate[i + 1], candidate[i + 2]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]] + self.dist_matrix[candidate[j + 1], candidate[0]]
            diff = weightafter - weightbefore
            return diff
        else:
            weightbefore = self.dist_matrix[current[i-1],current[i]] + self.dist_matrix[current[i+1],current[i+2]] + \
                           self.dist_matrix[current[j - 1], current[j]] + self.dist_matrix[current[j + 1], current[j + 2]]
            weightafter = self.dist_matrix[candidate[i-1],candidate[i]] + self.dist_matrix[candidate[i+1],candidate[i+2]] + \
                          self.dist_matrix[candidate[j - 1], candidate[j]] + self.dist_matrix[candidate[j + 1], candidate[j + 2]]
            diff = weightafter - weightbefore
            return diff
    def weightdiffSwapLocation(self,current,candidate,zeroBeforeIndex,zeroIndex,zeroAfterIndex):
        if zeroIndex == len(current) -1:
            weightbefore = self.dist_matrix[current[zeroBeforeIndex], current[zeroBeforeIndex + 1]] + \
                           self.dist_matrix[current[zeroIndex - 1], current[zeroIndex]] + self.dist_matrix[current[zeroIndex], current[0]] + \
                           self.dist_matrix[current[zeroAfterIndex - 1], current[zeroAfterIndex]]
        else:
            weightbefore= self.dist_matrix[current[zeroBeforeIndex],current[zeroBeforeIndex+1]] + \
                          self.dist_matrix[current[zeroIndex-1], current[zeroIndex]] + self.dist_matrix[current[zeroIndex], current[zeroIndex+1]] + \
                          self.dist_matrix[current[zeroAfterIndex-1], current[zeroAfterIndex]]
        if zeroAfterIndex == 0:
            weightafter = self.dist_matrix[candidate[zeroBeforeIndex], candidate[zeroBeforeIndex + 1]] + self.dist_matrix[candidate[zeroBeforeIndex + 1], candidate[zeroBeforeIndex + 2]] + \
                          self.dist_matrix[candidate[zeroIndex], candidate[zeroIndex + 1]] + self.dist_matrix[candidate[zeroIndex + 1], candidate[zeroIndex + 2]] + \
                          self.dist_matrix[candidate[zeroAfterIndex - 2], candidate[zeroAfterIndex-1]] + self.dist_matrix[candidate[zeroAfterIndex-1], candidate[zeroAfterIndex]]
        else:
            weightafter=  self.dist_matrix[candidate[zeroBeforeIndex],candidate[zeroBeforeIndex+1]] + self.dist_matrix[candidate[zeroBeforeIndex+1],candidate[zeroBeforeIndex+2]] + \
                          self.dist_matrix[candidate[zeroIndex], candidate[zeroIndex+1]] + self.dist_matrix[candidate[zeroIndex+1], candidate[zeroIndex+2]] + \
                          self.dist_matrix[candidate[zeroAfterIndex], candidate[zeroAfterIndex+1]]+self.dist_matrix[candidate[zeroAfterIndex+1], candidate[zeroAfterIndex+2]]
        diff = weightafter - weightbefore
        return diff

    def weightdiffDelSwapLocation(self,current,candidate,swapindices):
        ##print(swapindices)
        ##print(len(candidate))
        if swapindices[1] == len(candidate):
            weightbefore = self.dist_matrix[current[swapindices[0] - 1], current[swapindices[0]]] + self.dist_matrix[current[swapindices[0]], current[swapindices[0] + 1]] + \
                           self.dist_matrix[current[swapindices[1] - 1], current[swapindices[1]]] + self.dist_matrix[current[swapindices[1]], current[swapindices[1] + 1]] + \
                           self.dist_matrix[current[swapindices[2] - 1], current[swapindices[2]]] + self.dist_matrix[current[swapindices[2]], 0]
            weightafter = self.dist_matrix[candidate[swapindices[0] - 1], candidate[swapindices[0]]] + \
                          self.dist_matrix[candidate[swapindices[1] - 2], candidate[swapindices[1] - 1]] + \
                          self.dist_matrix[candidate[swapindices[1] - 1], candidate[0]] + \
                          self.dist_matrix[candidate[swapindices[2] - 2], candidate[0]]
        else:
            if swapindices[2] == len(current)-1:
                weightbefore = self.dist_matrix[current[swapindices[0] - 1], current[swapindices[0]]] + self.dist_matrix[current[swapindices[0]], current[swapindices[0] + 1]] + \
                               self.dist_matrix[current[swapindices[1] - 1], current[swapindices[1]]] + self.dist_matrix[current[swapindices[1]], current[swapindices[1] + 1]] + \
                               self.dist_matrix[current[swapindices[2] - 1], current[swapindices[2]]] + self.dist_matrix[current[swapindices[2]], 0]
                weightafter = self.dist_matrix[candidate[swapindices[0] - 1], candidate[swapindices[0]]] + \
                              self.dist_matrix[candidate[swapindices[1] - 2], candidate[swapindices[1] - 1]] + \
                              self.dist_matrix[candidate[swapindices[1] - 1], candidate[swapindices[1]]] + \
                              self.dist_matrix[candidate[swapindices[2] - 2], candidate[0]]
            else:
                weightbefore= self.dist_matrix[current[swapindices[0]-1],current[swapindices[0]]] + self.dist_matrix[current[swapindices[0]],current[swapindices[0]+1]] + \
                              self.dist_matrix[current[swapindices[1]-1],current[swapindices[1]]] + self.dist_matrix[current[swapindices[1]],current[swapindices[1]+1]] + \
                              self.dist_matrix[current[swapindices[2]-1],current[swapindices[2]]] + self.dist_matrix[current[swapindices[2]],current[swapindices[2]+1]]
                weightafter=  self.dist_matrix[candidate[swapindices[0]-1],candidate[swapindices[0]]] + \
                              self.dist_matrix[candidate[swapindices[1]-2],candidate[swapindices[1]-1]] + self.dist_matrix[candidate[swapindices[1]-1],candidate[swapindices[1]]] + \
                              self.dist_matrix[candidate[swapindices[2]-2],candidate[swapindices[2]-1]]
        diff = weightafter - weightbefore
        if abs(self.curr_weight + diff - self.weight(candidate)) > 3:
            ##print("DEL WRONG")
            time.sleep(1)
        return diff

    def acceptance_probability(self, candidate_weight):
        # (1 / (1 + np.exp(abs(candidate_weight - self.curr_weight) / self.temp)))
        try:
            seterr(all='raise')
            prob = np.exp(-np.abs((candidate_weight - self.curr_weight) / self.temp))
        except:
            prob=0
        # print('prob', prob)
        # print('candidate_weight', candidate_weight)
        # print('self.curr_weight', self.curr_weight)
        self.probability = prob
        return prob

    def accept(self, candidate, candidate_weight):
        '''
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        '''
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.temp *= self.alpha
                self.min_weight = candidate_weight
                self.best_solution = candidate
            return True
        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate
                #print(self.min_weight / candidate_weight)
                #self.temp *= self.min_weight / candidate_weight
                self.temp *= self.alpha
                return True
        return False
    def accept2(self, candidate, candidate_weight):
        '''
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        '''
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                #self.temp *= self.alpha
                self.min_weight = candidate_weight
                self.best_solution = candidate
            return True
        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate
                #print(self.min_weight / candidate_weight)
                self.temp *= self.min_weight / candidate_weight
                #self.temp *= self.alpha
                return True
        return False
    # ADDED
    # def endcriteria(self):
    #     if self.curr_weight > self.min_weight:
    #         self.counter += 1
    #     return self.counter < max(self.iteration / 10, self.stopping_iter)

    def endcriteria(self):
        # if time.time()-self.start > 20:
        #     return False
        if self.probability < 0.1:
            self.counter += 1
        if self.counter >=self.stopping_iter * (1+self.k/2): #FOR TOUR194 (1+self.k/2)
            self.k += 1
            print(self.counter3)
            if self.k < len(self.listlambadesired):
                self.temp = self.initialtemplist[self.k]
                self.counter = 0
                ##print("k",self.k)
                if self.min_weightlist[len(self.min_weightlist)-1] == self.min_weight and self.curr_weight < self.initial_weight:
                    self.counter3 += 1
                    if self.counter3 > 5 :
                        # self.timer.append(["candidate1",self.candidate1])
                        # self.timer.append(["candidate2", self.candidate2])
                        # self.timer.append(["candidate3", self.candidate3])
                        # self.timer.append(["candidate4", self.candidate4])
                        # self.timer.append(["candidate5", self.candidate5])
                        # self.timer.append(["sorting", self.sort])
                        # self.timer.append(["min", self.min])
                        # self.timer.append(["weight", self.timeweight])
                        # self.timer.append(["weightdiffswap", self.timeweightdiffswap])
                        # self.timer.append(["weightdiffdisplacement", self.timeweightdiffdisplacement])
                        # self.timer.append(["weightdiffinsert", self.timeweightdiffinsert])
                        # self.timer.append(["weightdiffinverse", self.timeweightdiffinverse])
                        ##print(self.best_solution)
                        return False
                elif self.min_weightlist[len(self.min_weightlist)-1] != self.min_weight:
                    self.counter3 = 0;
                self.min_weightlist.append(self.min_weight)
                print("k",self.k)
                print("minweight", self.min_weight)
                print("currweight", self.curr_weight)
                return True
            else:
                # self.timer.append(["candidate1", self.candidate1])
                # self.timer.append(["candidate2", self.candidate2])
                # self.timer.append(["candidate3", self.candidate3])
                # self.timer.append(["candidate4", self.candidate4])
                # self.timer.append(["candidate5", self.candidate5])
                # self.timer.append(["sorting", self.sort])
                # self.timer.append(["min", self.min])
                # self.timer.append(["weight", self.timeweight])
                # self.timer.append(["weightdiffswap", self.timeweightdiffswap])
                # self.timer.append(["weightdiffdisplacement", self.timeweightdiffdisplacement])
                # self.timer.append(["weightdiffinsert", self.timeweightdiffinsert])
                # self.timer.append(["weightdiffinverse", self.timeweightdiffinverse])
                # print(self.best_solution)
                return False
        return True
    def endcriteria2(self):
        if time.time()-self.start > 300:
             return False
        if self.probability < 0.1:
            self.counter += 1
            self.counter += 1
        if self.counter >=self.stopping_iter * (1+self.k/2): #FOR TOUR194 (1+self.k/2)
            self.k += 1
            print("counter: ",self.counter3)
            if self.k < len(self.listlambadesired):
                self.temp = self.initialtemplist[self.k]
                self.counter = 0
                ##print("k",self.k)
                if self.min_weightlist[len(self.min_weightlist)-1] == self.min_weight and self.curr_weight < self.initial_weight:
                    self.counter3 += 1
                    if self.counter3 > 5 :
                        # self.timer.append(["candidate1",self.candidate1])
                        # self.timer.append(["candidate2", self.candidate2])
                        # self.timer.append(["candidate3", self.candidate3])
                        # self.timer.append(["candidate4", self.candidate4])
                        # self.timer.append(["candidate5", self.candidate5])
                        # self.timer.append(["sorting", self.sort])
                        # self.timer.append(["min", self.min])
                        # self.timer.append(["weight", self.timeweight])
                        # self.timer.append(["weightdiffswap", self.timeweightdiffswap])
                        # self.timer.append(["weightdiffdisplacement", self.timeweightdiffdisplacement])
                        # self.timer.append(["weightdiffinsert", self.timeweightdiffinsert])
                        # self.timer.append(["weightdiffinverse", self.timeweightdiffinverse])
                        ##print(self.best_solution)
                        return False
                else:
                    self.counter3 = 0;
                self.min_weightlist.append(self.min_weight)
                print("k",self.k)
                print("minweight", self.min_weight)
                print("currweight", self.curr_weight)
                return True
            else:
                # self.timer.append(["candidate1", self.candidate1])
                # self.timer.append(["candidate2", self.candidate2])
                # self.timer.append(["candidate3", self.candidate3])
                # self.timer.append(["candidate4", self.candidate4])
                # self.timer.append(["candidate5", self.candidate5])
                # self.timer.append(["sorting", self.sort])
                # self.timer.append(["min", self.min])
                # self.timer.append(["weight", self.timeweight])
                # self.timer.append(["weightdiffswap", self.timeweightdiffswap])
                # self.timer.append(["weightdiffdisplacement", self.timeweightdiffdisplacement])
                # self.timer.append(["weightdiffinsert", self.timeweightdiffinsert])
                # self.timer.append(["weightdiffinverse", self.timeweightdiffinverse])
                # print(self.best_solution)
                return False
        return True
    def anneal(self):
        ##print('Initial weight: ', self.curr_weight)
        tbegin = time.time()
        timeIteration = time.time()
        t0 = time.time()
        self.initialtemplist=self.getInitialTemp()
        t1 = time.time()
        self.timer.append(["Initialtemp",t1-t0])
        self.temp = self.initialtemplist[0]
        self.initialtemp = self.temp
        while self.endcriteria():
            candidatelist = []
            candidateweightlist = []
            for _ in range(self.k+1):
                # inverse
                t0 = time.time()
                candidate = list(self.curr_solution)
                i = random.randint(1, self.sample_size - 2)
                j = random.randint(2, self.sample_size-i)
                candidate[i: (i + j)] = reversed(candidate[i: (i + j)])
                candidatelist.append(candidate)
                candidateweightlist.append(self.weightdiffinverse(self.curr_solution,candidate,i,i+j-1))
                t1 = time.time()
                self.candidate1 += t1-t0
                # swap
                # t0 = time.time()
                # candidate2 = list(self.curr_solution)
                # j = random.randint(1, self.sample_size-1)
                # i = random.randint(1, self.sample_size-1)
                # candidate2[j], candidate2[i] = candidate2[i], candidate2[j]
                # candidatelist.append(candidate2)
                # candidateweightlist.append(self.weightdiffswap(self.curr_solution, candidate2, j, i))
                # t1 = time.time()
                # self.candidate2 += t1 - t0
                # # insert
                # t0 = time.time()
                # candidate3 = list(self.curr_solution)
                # j = random.randint(1, self.sample_size - 1)
                # i = random.randint(1, self.sample_size - 2)
                # city = candidate3.pop(j)
                # while i == j:
                #     i = random.randint(1, self.sample_size - 2)
                # if i < j:
                #     candidate3.insert(i, city)
                # else:
                #     if i - 1 == j:
                #         i += 1
                #     candidate3.insert(i-1,city)
                # candidatelist.append(candidate3)
                # candidateweightlist.append(self.weightdiffinsert(self.curr_solution, candidate3, i, j))
                # t1 = time.time()
                # self.candidate3 += t1 - t0
                # #Displacement
                # t0 = time.time()
                # candidate4 = list(self.curr_solution)
                # i =  random.randint(1, self.sample_size - 2)
                # j =  random.randint(i+1, self.sample_size - 1)
                # r = random.randint(1, self.sample_size-(j-i)-1)
                # if r < j and r < i:
                #     candidate4 = candidate4[0:r] + candidate4[i:j + 1] + candidate4[r:i] + candidate4[j + 1:self.sample_size]
                # elif r != i:
                #     candidate4 = candidate4[0:i] + candidate4[j + 1:r + j - i + 1] + candidate4[i:j + 1] + candidate4[r + j - i + 1:self.sample_size]
                # candidatelist.append(candidate4)
                # candidateweightlist.append(self.weightdiffDisplacement(self.curr_solution, candidate4, i, j, r))
                # t1 = time.time()
                # self.candidate4 += t1 - t0
                # # Bigger Swap
                # t0 = time.time()
                # candidate5 = list(self.curr_solution)
                # i = 0
                # j = 1
                # while abs(i-j) == 1:
                #     i = random.randint(1, self.sample_size - 2)
                #     j = random.randint(1, self.sample_size - 2)
                # candidate5[i], candidate5[j] = candidate5[j], candidate5[i]
                # candidate5[i + 1], candidate5[j + 1] = candidate5[j + 1], candidate5[i + 1]
                # candidatelist.append(candidate5)
                # candidateweightlist.append(self.weightdiffBigswap(self.curr_solution, candidate5, i, j))
                # t1 = time.time()
                # self.candidate5 += t1 - t0
            t0 = time.time()
            index_min = np.argmin(candidateweightlist)
            best = candidatelist[index_min]
            t1 = time.time()
            self.min += t1-t0
            self.accept(best,self.curr_weight+candidateweightlist[index_min])
            self.templist.append(self.temp)
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)
            timeIterationEnd = time.time()
            self.timeiterations.append(timeIterationEnd-timeIteration)
        ##print('Number of iteration: ', self.iteration)
        ##print('Minimum weight: ', self.min_weight)
        ##print('Improvement: ', round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')
        ##print('best tour:',self.best_solution)
        tend = time.time()
        self.timer.append(["SA: ", tend - tbegin])
        ##print('time:', self.timer)
        return self.best_solution

    def illegal(self,candidate,quantities,maxquantity):
        "Test on quantities"
        if candidate == 0:
            return True
        if candidate == self.curr_solution:
            return False
        sum=0
        for x in candidate:
            sum += quantities[x]
            if sum > maxquantity:
                return True
            if x == 0:
                sum=0
        return False


    def annealDepot(self,quantities,maxquantity,depotsolution):
        self.sample_size = len(depotsolution)
        self.curr_solution = depotsolution
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]
        self.curr_weight = self.weight(self.curr_solution)
        self.old_weight = self.curr_weight
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]
        self.min_weightlist = [self.min_weight]
        tbegin = time.time()
        timeIteration = time.time()
        t0 = time.time()
        self.initialtemplist=self.getInitialTemp()
        t1 = time.time()
        ##print('Initial weight: ', self.curr_weight)
        self.timer.append(["Initialtemp",t1-t0])
        self.temp = self.initialtemplist[0]
        self.initialtemp = self.temp

        while self.endcriteria2():

            candidatelist = []
            candidateweightlist = []
            for _ in range(self.k+1):
                # inverse
                t0 = time.time()
                candidate = 0
                candidate = list(self.curr_solution)
                i = random.randint(1, self.sample_size - 2)
                j = random.randint(2, self.sample_size-i)
                candidate[i: (i + j)] = reversed(candidate[i: (i + j)])
                if self.illegal(candidate, quantities, maxquantity):
                    candidate=self.curr_solution
                candidatelist.append(candidate)
                if (candidate != self.curr_solution):
                    candidateweightlist.append(self.weightdiffinverse(self.curr_solution,candidate,i,i+j-1))
                else:
                    candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate1 += t1-t0
                # swap
                t0 = time.time()
                candidate2 = 0
                candidate2 = list(self.curr_solution)
                j = random.randint(1, self.sample_size-1)
                i = random.randint(1, self.sample_size-1)
                candidate2[j], candidate2[i] = candidate2[i], candidate2[j]
                if self.illegal(candidate2, quantities, maxquantity):
                        candidate2 = self.curr_solution
                candidatelist.append(candidate2)
                if (candidate2 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffswap(self.curr_solution, candidate2, j, i))
                else:
                    candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate2 += t1 - t0
                # insert
                t0 = time.time()
                candidate3 = 0
                candidate3 = list(self.curr_solution)
                j = random.randint(1, self.sample_size - 1)
                i = random.randint(1, self.sample_size - 2)
                city = candidate3.pop(j)
                while i == j:
                     i = random.randint(1, self.sample_size - 2)
                if i < j:
                    candidate3.insert(i, city)
                else:
                    if i - 1 == j:
                        i += 1
                    candidate3.insert(i-1,city)
                if self.illegal(candidate3, quantities, maxquantity):
                    candidate3 = self.curr_solution
                candidatelist.append(candidate3)
                if (candidate3 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffinsert(self.curr_solution, candidate3, i, j))
                else:
                    candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate3 += t1 - t0
                #Displacement
                t0 = time.time()
                candidate4 = 0
                candidate4 = list(self.curr_solution)
                i =  random.randint(1, self.sample_size - 2)
                j =  random.randint(i+1, self.sample_size - 1)
                r = random.randint(1, self.sample_size-(j-i)-1)
                if r < j and r < i:
                    candidate4 = candidate4[0:r] + candidate4[i:j + 1] + candidate4[r:i] + candidate4[j + 1:self.sample_size]
                elif r != i:
                    candidate4 = candidate4[0:i] + candidate4[j + 1:r + j - i + 1] + candidate4[i:j + 1] + candidate4[r + j - i + 1:self.sample_size]
                if self.illegal(candidate4, quantities, maxquantity):
                        candidate4 = self.curr_solution
                candidatelist.append(candidate4)
                if (candidate4 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffDisplacement(self.curr_solution, candidate4, i, j, r))
                else:
                    candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate4 += t1 - t0
                # Bigger Swap
                t0 = time.time()
                candidate5 = 0
                candidate5 = list(self.curr_solution)
                i = 0
                j = 1
                while abs(i-j) == 1:
                    i = random.randint(1, self.sample_size - 2)
                    j = random.randint(1, self.sample_size - 2)
                candidate5[i], candidate5[j] = candidate5[j], candidate5[i]
                candidate5[i + 1], candidate5[j + 1] = candidate5[j + 1], candidate5[i + 1]
                if self.illegal(candidate5, quantities, maxquantity):
                    candidate5 = self.curr_solution
                candidatelist.append(candidate5)
                if (candidate5 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffBigswap(self.curr_solution, candidate5, i, j))
                else:
                    candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate5 += t1 - t0
            t0 = time.time()
            index_min = np.argmin(candidateweightlist)
            best = candidatelist[index_min]
            t1 = time.time()
            self.min += t1-t0
            self.accept(best,self.curr_weight+candidateweightlist[index_min])
            self.templist.append(self.temp)
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)
            timeIterationEnd = time.time()
            self.timeiterations.append(timeIterationEnd-timeIteration)
        ##print('Number of iteration: ', self.iteration)
        ##print('Minimum weight: ', self.min_weight)
        ##print('Improvement: ', round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')
        ##print('best tour:',self.best_solution)
        tend = time.time()
        self.timer.append(["SA: ", tend - tbegin])
        ##print('time:', self.timer)
        return self.best_solution

    def illegalswap(self,candidate,quantities,maxquantity):
        "Test on quantities"
        if candidate == self.curr_solution:
            return False
        sum=0
        swap=0
        swapnumber= 0
        for x in range(len(candidate)):
            if candidate[x] < self.oldsample_size:
                sum += quantities[candidate[x]]
            if sum > maxquantity:
                ###print("bigger than maxquantity")
                return True
            if candidate[x] == 0:
                if swap == 0:
                    sum=0
                else:
                    swap=0
            if candidate[x] > self.oldsample_size-1:
                swap += 1
                if swap == 1 and candidate[x-1] != 0:
                    ###print("Previous one not 0")
                    return True
                elif swap == 1:
                    swapnumber=candidate[x]
                if swap == 2 and candidate[x] != swapnumber:
                    ###print(swapnumber)
                    ###print(candidate[x])
                    ###print("Not same number")
                    return True
                elif swap == 2:
                    sum = 0
                if swap == 3 and candidate[x] != swapnumber:
                    ###print(swapnumber)
                    ###print(candidate[x])
                    ###print("Not same number")
                    return True
                elif swap == 3:
                    if x == len(candidate) - 1 and candidate[0]==0:
                        swap=0
                    elif candidate[x+1] == 0:
                        swap=0
                    else:
                        ###print("Next one not 0")
                        return True
        return False
    def searchSwapLocation(self,candidate,zerocoordinates,indexinlist,quantities,maxquantity):
        zeroIndex= zerocoordinates[indexinlist]
        candidatelist = []
        weightlist = []
        lengthswapindex = []
        swapindexes =list(range(self.oldsample_size,self.dist_matrix.shape[0]))
        lastzero = zerocoordinates[-1]
        for swapIndex in range(self.oldsample_size, self.dist_matrix.shape[0]):
            lengthswapindex.append(self.dist_matrix[swapIndex,candidate[zeroIndex-1]])
        swapindexes = [x for _, x in sorted(zip(lengthswapindex, swapindexes))]
        swapindexes = swapindexes[0:round(len(swapindexes)/5)]
        for swapIndex in swapindexes:
            newcandidate = list(candidate)
            if zeroIndex == lastzero:
                newcandidate.insert(len(newcandidate), swapIndex)
            else:
                newcandidate.insert(zerocoordinates[indexinlist+1],swapIndex)
            newcandidate[zeroIndex] = swapIndex
            newcandidate.insert(zerocoordinates[indexinlist - 1]+1, swapIndex)
            candidatelist.append(newcandidate)
            if indexinlist == len(zerocoordinates) - 1:
                weightlist.append(self.weightdiffSwapLocation(self.curr_solution, newcandidate, zerocoordinates[indexinlist - 1],zeroIndex, 0))
            else:
                weightlist.append(self.weightdiffSwapLocation(self.curr_solution, newcandidate, zerocoordinates[indexinlist - 1],zeroIndex, zerocoordinates[indexinlist + 1]))
        bestIndex = np.argmin(weightlist)
        bestcandidate = candidatelist[bestIndex]
        weight = weightlist[bestIndex]
        return [bestcandidate,weight]
    def annealSwapLocations(self, quantities, maxquantity, depotsolution,oldsamplesize):
        self.sample_size = len(depotsolution)
        self.oldsample_size= oldsamplesize
        self.curr_solution = depotsolution
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]
        self.curr_weight = self.weight(self.curr_solution)
        self.old_weight = self.curr_weight
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]
        self.min_weightlist = [self.min_weight]
        tbegin = time.time()
        timeIteration = time.time()
        t0 = time.time()
        self.initialtemplist = self.getInitialTemp()
        t1 = time.time()
        ##print('Initial weight: ', self.curr_weight)
        self.timer.append(["Initialtemp", t1 - t0])
        self.temp = self.initialtemplist[0]
        self.initialtemp = self.temp

        while self.endcriteria2():
            if self.sample_size != len(self.curr_solution):
                ##print("ERROR")
                ##print(self.sample_size)
                ##print(len(self.curr_solution))
                time.sleep(1)
            candidatelist = []
            candidateweightlist = []
            for _ in range(self.k+1):
                # inverse
                t0 = time.time()
                candidate = 0
                candidate = list(self.curr_solution)
                i = random.randint(1, self.sample_size - 2)
                j = random.randint(2, self.sample_size-i)
                candidate[i: (i + j)] = reversed(candidate[i: (i + j)])
                if self.illegalswap(candidate, quantities, maxquantity):
                    candidate=self.curr_solution
                candidatelist.append(candidate)
                if (candidate != self.curr_solution):
                    candidateweightlist.append(self.weightdiffinverse(self.curr_solution,candidate,i,i+j-1))
                else:
                    candidateweightlist.append(10000)
                t1 = time.time()
                self.candidate1 += t1-t0
                # swap
                t0 = time.time()
                candidate2 = 0
                candidate2 = list(self.curr_solution)
                j = random.randint(1, self.sample_size-1)
                i = random.randint(1, self.sample_size-1)
                candidate2[j], candidate2[i] = candidate2[i], candidate2[j]
                if self.illegalswap(candidate2, quantities, maxquantity):
                        candidate2 = self.curr_solution
                candidatelist.append(candidate2)
                if (candidate2 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffswap(self.curr_solution, candidate2, j, i))
                else:
                    candidateweightlist.append(10000)
                t1 = time.time()
                self.candidate2 += t1 - t0
                # insert
                t0 = time.time()
                candidate3 = 0
                candidate3 = list(self.curr_solution)
                j = random.randint(1, self.sample_size - 1)
                i = random.randint(1, self.sample_size - 2)
                city = candidate3.pop(j)
                while i == j:
                     i = random.randint(1, self.sample_size - 2)
                if i < j:
                    candidate3.insert(i, city)
                else:
                    if i - 1 == j:
                        i += 1
                    candidate3.insert(i-1,city)
                if self.illegalswap(candidate3, quantities, maxquantity):
                    candidate3 = self.curr_solution
                candidatelist.append(candidate3)
                if (candidate3 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffinsert(self.curr_solution, candidate3, i, j))
                else:
                    candidateweightlist.append(10000)
                t1 = time.time()
                self.candidate3 += t1 - t0
                #Displacement
                t0 = time.time()
                candidate4 = 0
                candidate4 = list(self.curr_solution)
                i =  random.randint(1, self.sample_size - 2)
                j =  random.randint(i+1, self.sample_size - 1)
                r = random.randint(1, self.sample_size-(j-i)-1)
                if r < j and r < i:
                    candidate4 = candidate4[0:r] + candidate4[i:j + 1] + candidate4[r:i] + candidate4[j + 1:self.sample_size]
                elif r != i:
                    candidate4 = candidate4[0:i] + candidate4[j + 1:r + j - i + 1] + candidate4[i:j + 1] + candidate4[r + j - i + 1:self.sample_size]
                if self.illegalswap(candidate4, quantities, maxquantity):
                        candidate4 = self.curr_solution
                candidatelist.append(candidate4)
                if (candidate4 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffDisplacement(self.curr_solution, candidate4, i, j, r))
                else:
                    candidateweightlist.append(10000)
                t1 = time.time()
                self.candidate4 += t1 - t0
                # Bigger Swap
                t0 = time.time()
                candidate5 = 0
                candidate5 = list(self.curr_solution)
                i = 0
                j = 1
                while abs(i-j) == 1:
                    i = random.randint(1, self.sample_size - 2)
                    j = random.randint(1, self.sample_size - 2)
                candidate5[i], candidate5[j] = candidate5[j], candidate5[i]
                candidate5[i + 1], candidate5[j + 1] = candidate5[j + 1], candidate5[i + 1]
                if self.illegalswap(candidate5, quantities, maxquantity):
                    candidate5 = self.curr_solution
                candidatelist.append(candidate5)
                if (candidate5 != self.curr_solution):
                    candidateweightlist.append(self.weightdiffBigswap(self.curr_solution, candidate5, i, j))
                else:
                    candidateweightlist.append(10000)
                t1 = time.time()
                self.candidate5 += t1 - t0
                # USE Swap Location
                ##print("Swap location")
                t0 = time.time()
                candidate6 = 0
                    #STEP 1 SEARCH RANDOM ZERO (NOT FIRST ONE)
                realzerocoordinates = []
                zerocoordinates = []
                for i in np.arange(0, len(self.curr_solution)):
                    if i != len(self.curr_solution)-1:
                        if self.curr_solution[i] == 0 and self.curr_solution[i-1]<self.oldsample_size-1 and self.curr_solution[i+1]<self.oldsample_size-1 and i != 0:
                            realzerocoordinates.append(i)
                        if self.curr_solution[i] == 0:
                            zerocoordinates.append(i)
                    else:
                        if self.curr_solution[i] == 0 and self.curr_solution[i-1]<self.oldsample_size-1 and i != 0:
                            realzerocoordinates.append(i)
                        if self.curr_solution[i] == 0:
                            zerocoordinates.append(i)
                candidate6 = list(self.curr_solution)
                zeroIndex=0
                if len(realzerocoordinates) > 0:
                    while zeroIndex == 0:
                        zeroIndex=random.choice(realzerocoordinates)
                    indexinlist= zerocoordinates.index(zeroIndex)
                            #SEARCH RANDOM SWAP LOCATION => not good enough so we need to find local ones
                    bestlist=self.searchSwapLocation(candidate6,zerocoordinates,indexinlist,quantities,maxquantity)
                    candidate6=bestlist[0]
                        # swapIndex= random.randint(self.oldsample_size,self.dist_matrix.shape[0]-1)
                        # if indexinlist == len(zerocoordinates)-1:
                        #     candidate6.insert(len(candidate6), swapIndex)
                        # else:
                        #     candidate6.insert(zerocoordinates[indexinlist+1],swapIndex)
                        # candidate6[zeroIndex] = swapIndex
                        # candidate6.insert(zerocoordinates[indexinlist - 1]+1, swapIndex)
                    if self.illegalswap(candidate6, quantities, maxquantity):
                             candidate6 = self.curr_solution
                             bestlist = [self.curr_solution,100000]
                    candidatelist.append(bestlist[0])
                    candidateweightlist.append(bestlist[1])
                else:
                    candidatelist.append(self.curr_solution)
                    candidateweightlist.append(100000)
                # if (len(candidate6)+2 == len(self.curr_solution)):
                #     if indexinlist == len(zerocoordinates) - 1:
                #         candidateweightlist.append(self.weightdiffSwapLocation(self.curr_solution, candidate6,zerocoordinates[indexinlist - 1],zeroIndex,0))
                #     else:
                #         candidateweightlist.append(self.weightdiffSwapLocation(self.curr_solution, candidate6,zerocoordinates[indexinlist-1],zeroIndex,zerocoordinates[indexinlist+1]))
                # else:
                #     candidateweightlist.append(100000)
                t1 = time.time()
                self.candidate6 += t1 - t0
                # DELETE Swap Location
                t0 = time.time()
                candidate7 = 0
                swapcoordinates = []
                for i in np.arange(1, len(self.curr_solution)):
                    if self.curr_solution[i] > self.oldsample_size - 1:
                        swapcoordinates.append(i)
                if len(swapcoordinates) > 0:
                    candidate7 = list(self.curr_solution)
                    swapIndex = random.choice(swapcoordinates)
                    swapindices = []
                    for i in range(len(swapcoordinates)):
                        if candidate7[swapcoordinates[i]] == candidate7[swapIndex]:
                            swapindices.append(swapcoordinates[i])
                    candidate7.pop(swapindices[2])
                    candidate7[swapindices[1]] = 0
                    candidate7.pop(swapindices[0])
                    if self.illegalswap(candidate7, quantities, maxquantity):
                        candidate7 = self.curr_solution
                    candidatelist.append(candidate7)
                    if (candidate7 != self.curr_solution):
                        candidateweightlist.append(self.weightdiffDelSwapLocation(self.curr_solution,candidate7,swapindices))
                    else:
                        candidateweightlist.append(100000)
                    t1 = time.time()
                    self.candidate7 += t1 - t0
                else:
                    candidatelist.append(self.curr_solution)
                    candidateweightlist.append(100000)
               # print("Done mutation")
            t0 = time.time()
            index_min = np.argmin(candidateweightlist)
            t1 = time.time()
            best = candidatelist[index_min]
            self.min += t1 - t0
            if (self.accept2(best, self.curr_weight + candidateweightlist[index_min])):
                if index_min != 0:
                    if index_min % 7 == 5:
                        ##print("+2")
                        self.sample_size= self.sample_size + 2
                    if index_min % 7 == 6:
                        ##print("-2")
                        self.sample_size = self.sample_size - 2
            self.templist.append(self.temp)
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)
            timeIterationEnd = time.time()
            self.timeiterations.append(timeIterationEnd - timeIteration)
        print('Number of iteration: ', self.iteration)
        print('Minimum weight: ', self.min_weight)
        print('Improvement: ', round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')
        print('best tour:', self.best_solution)
        tend = time.time()
        self.timer.append(["SA: ", tend - tbegin])
        ##print('time:', self.timer)
        return self.best_solution

   # def animateSolutions(self):
    #    animated_visualizer.animateTSP(self.solution_history, self.coords)

    def plotLearningWeight(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()

    def plotLearningWeightTime(self):
        plt.plot([i for i in self.timeiterations], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Time')
        plt.show()

    def plotLearningTemp(self):
        plt.plot([i for i in range(len(self.templist))], self.templist)
        #plt.xscale("log")
        line_init = plt.axhline(y=self.initialtemp, color='r', linestyle='--')
        plt.legend([line_init], ['Initial Temp'])
        plt.ylabel('Temperature')
        plt.xlabel('Iteration')
        plt.show()

    def plotLearningTime(self):
        objects = ('inverse', 'swap', 'insert', 'displacement', 'big swap')
        y_pos = np.arange(len(objects))
        performance = [self.candidate1,self.candidate2,self.candidate3,self.candidate4,self.candidate5]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Time spent calculating (seconds)')
        plt.title('Different mutations operators')
        plt.show()
    def plotDepottour(self,depottour,pairs,xcoordsstring,ycoordsstring,xdepotstring,ydepotstring,xswapstring,yswapstring):
        xcoords = []
        ycoords = []
        xdepot = []
        ydepot = []
        xswap = []
        yswap = []
        for x in xcoordsstring:
            xcoords.append(float(x))
        for y in ycoordsstring:
            ycoords.append(float(y))
        for x in xdepotstring:
            xdepot.append(float(x))
        for y in ydepotstring:
            ydepot.append(float(y))
        for x in xswapstring:
            xswap.append(float(x))
        for y in yswapstring:
            yswap.append(float(y))
        plt.scatter(xcoords, ycoords, color="blue")
        plt.scatter(xdepot, ydepot, color="red")
        plt.scatter(xswap, yswap, color="lime")
        data = []
        for city in depottour:
            data.append(pairs[city])
        data.append(pairs[0])
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1])
        plt.title('Tour consisting of all customers')
        plt.show()

    def showCoordinates(self,xcoordsstring,ycoordsstring,xdepotstring,ydepotstring,xswapstring,yswapstring):
        xcoords=[]
        ycoords=[]
        xdepot=[]
        ydepot=[]
        xswap=[]
        yswap=[]
        for x in xcoordsstring:
            xcoords.append(float(x))
        for y in ycoordsstring:
            ycoords.append(float(y))
        for x in xdepotstring:
            xdepot.append(float(x))
        for y in ydepotstring:
            ydepot.append(float(y))
        for x in xswapstring:
            xswap.append(float(x))
        for y in yswapstring:
            yswap.append(float(y))
        plt.scatter(xcoords, ycoords,color="blue")
        plt.scatter(xdepot, ydepot,color="red")
        plt.scatter(xswap,yswap,color="lime")
        plt.title('Coordinates of all customers and depot')
        plt.show()

    # Sometime lambadesired rises instead of going lower
    def InitialTemperature(self, lambadesired):
        lambatemp = 100
        randomtours = self.initialize()
        posTransitions = []
        for tour in randomtours:
            initialweight = self.weight(tour)
            posTransitions.append(self.getPosTransition(tour, initialweight))
        temperature = self.getStartTemp(randomtours, posTransitions, lambadesired)
        minweight = min([self.weight(tour) for tour in posTransitions])
        while abs(lambatemp - lambadesired) > 0.05:
            part = sum([np.exp((-self.weight(tour) / temperature) + (minweight / temperature)) for tour in randomtours])
            participle = sum([np.exp((-self.weight(tour) / temperature) + (minweight / temperature)) for tour in posTransitions])
            ##print('part', part)
            ##print('participle', participle)
            lambatemp = part / participle
            ##print('lambatemp', lambatemp)
            previoustemp = temperature
            ##print((np.log(lambatemp) / np.log(lambadesired)))
            temperature = previoustemp * (np.log(lambatemp) / np.log(lambadesired))
            ##print(temperature)
        if np.isnan(temperature):
            return self.InitialTemperature(lambadesired)
        return temperature

    # already calculate weight in it but not stored
    def getPosTransition(self, initialCandidate, initialweight):
        for counter in range(25):
            k = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - k)
            postrans = list(initialCandidate)
            postrans[i: (i + k)] = reversed(postrans[i: (i + k)])
            weight = SimulatedAnnealing.weight(self, postrans)
            if weight < initialweight:
                return postrans
        return initialCandidate

    def initialize(self):
        """Random init"""
        tours = [np.random.permutation(range(1, self.diffsample)) for i in range(200)]
        tours = np.array([np.concatenate((np.array([0]), tour)) for tour in tours])
        return tours

    def getStartTemp(self, randomtours, posTransitions, lambadesired):
        randomweight = sum([self.weight(tour) for tour in randomtours])
        posweight = sum([self.weight(tour) for tour in posTransitions])
        starttemp = abs((posweight - randomweight) / (200 * np.log(lambadesired)))
        ##print('starttemp', starttemp)
        return starttemp
    def getInitialTemp(self):
        randomtours = self.initialize()
        posTransitions = []
        for tour in randomtours:
            initialweight = self.weight(tour)
            posTransitions.append(self.getPosTransition(tour, initialweight))
        temperaturelist = []
        randomweight = sum([self.weight(tour) for tour in randomtours])
        posweight = sum([self.weight(tour) for tour in posTransitions])
        average = abs(posweight - randomweight) / 200
        for k in range(len(self.listlambadesired)):
            temperature = -average/np.log(self.listlambadesired[k])
            temperaturelist.append(temperature)
        return temperaturelist;
    def simpleDepotStops(self,besttour,quantities,maxquantity):
        sum = 0
        index =0
        while index in range(len(besttour)):
            sum += quantities[besttour[index]]
            if sum > maxquantity:
                sum=0
                besttour.insert(index,0)
            index += 1
        if self.illegal(besttour,quantities,maxquantity):
            print("WRRROOONG")
        return besttour
    def illegal(self,candidate,quantities,maxquantity):
        "Test on quantities"
        if candidate == 0:
            return True
        if candidate == self.curr_solution:
            return False
        sum=0
        for x in candidate:
            sum += quantities[x]
            if sum > maxquantity:
                return True
            if x == 0:
                sum=0
        return False
def main():
    temp = 1000
    stopping_temp = 0.00000001
    alpha = 0.95
    #stopping_iter = 300  # tour29
    stopping_iter = 50  # tour194
    #stopping_iter = 300 # tour929
    matrix = np.loadtxt("tour194.csv", delimiter=",")
    ##print(matrix)
    # algorithm = SimulatedAnnealing(matrix, temp, alpha, stopping_temp, stopping_iter,listlambadesired)
    # print(algorithm.anneal())
    # # algorithm.animateSolutions()
    # algorithm.plotLearningWeightTime()
    # algorithm.plotLearningWeight()
    # algorithm.plotLearningTemp()
    # algorithm.plotLearningTime()


def test(alpha,stopping_iter,DATFILE,location,lambda1,lambda2):
    import pandas
    df = pandas.read_csv(location,names=['LOCATION_TYPE','LOCATION_ID','POST_CODE','CITY','X_COORD','Y_COORD','QUANTITY','TRAIN_POSSIBLE','SERVICE_TIME [s]'],sep=';')
    x_coords= (df.loc[df['LOCATION_TYPE'] == 'CUSTOMER', 'X_COORD']).to_numpy()
    y_coords= (df.loc[df['LOCATION_TYPE'] == 'CUSTOMER', 'Y_COORD']).to_numpy()
    x_depot= (df.loc[df['LOCATION_TYPE'] == 'DEPOT', 'X_COORD']).to_numpy()
    y_depot= (df.loc[df['LOCATION_TYPE'] == 'DEPOT', 'Y_COORD']).to_numpy()
    x_swap= (df.loc[df['LOCATION_TYPE'] == 'SWAP_LOCATION', 'X_COORD']).to_numpy()
    y_swap= (df.loc[df['LOCATION_TYPE'] == 'SWAP_LOCATION', 'Y_COORD']).to_numpy()
    quantitiesstring = (df.loc[df['LOCATION_TYPE'] == 'CUSTOMER', 'QUANTITY']).to_numpy()
    df2 = pandas.read_csv('Fleet.csv',names=['TYPE','CAPACITY','COSTS [MU/km]','COSTS [MU/h]','COSTS [MU/USAGE]','OPERATING_TIME [s]'], sep=';')
    maxquantity = (df2.loc[df2['TYPE'] == 'SWAP_BODY', 'CAPACITY']).to_numpy()
    maxquantity = float(maxquantity[0])
    ##print('X_COORD:',x_coords)
    ##print('Y_COORD:',y_coords)
    ##print('X-DEPOT:',x_depot)
    ##print('Y-DEPOT:',y_depot)
    ##print('Quantities:',quantitiesstring)
    ##print('Maxquantity:',maxquantity)
    quantities = [0]
    for quantity in quantitiesstring:
        quantities.append(float(quantity))
    pairs= []
    matrix = []
    pairs.append([float(x_depot[0]),float(y_depot[0])])
    for k in range(len(x_coords)):
        pairs.append([float(x_coords[k]),float(y_coords[k])])
    for k2 in range(len(pairs)):
        row=[]
        for k3 in range(len(pairs)):
            row.append(((float(pairs[k2][0])-float(pairs[k3][0]))**2+(float(pairs[k2][1])-float(pairs[k3][1]))**2)**0.5)
        matrix.append(row)
    matrix = np.array(matrix)
    print(matrix)
    "BEGIN OF VARIABLES"
    listlambadesired = list(np.linspace(0.1, lambda1/20, 50))
    listlambadesired2 = list(np.linspace(0.1, lambda2/20, 50))
    listlambadesired.reverse()
    listlambadesired2.reverse()
    #alpha = 0.95
    ## stopping_iter = 300  # tour29
    #stopping_iter = 50  # tour194
    # stopping_iter = 300 # tour929
    "END OF VARIABLES"
    algorithm = SimulatedAnnealing(matrix.copy(),alpha,stopping_iter, listlambadesired)
    print("Starting besttour")
    besttour = algorithm.anneal()
    print("besttour: ",besttour)
    depottour = algorithm.simpleDepotStops(besttour.copy(),quantities,maxquantity)
    algorithm2 = SimulatedAnnealing(matrix.copy(),alpha,stopping_iter,listlambadesired2)
    print("Starting betterdepottour")
    if (algorithm2.illegal(depottour,quantities,maxquantity)):
        print("ILLEGAL FROM START")
        time.sleep(20)
    betterdepottour = algorithm2.annealDepot(quantities,maxquantity,depottour.copy())
    print("betterdepottour: ",betterdepottour)

    "CALCULATE NEW DISTANCEMATRIX NOW WITH SWAPLOCATIONS, SWAPLOCATIONS SHOULD BE ADDED LAST TO PAIRS"
    newmatrix = []
    for k in range(len(x_swap)):
        pairs.append([float(x_swap[k]),float(y_swap[k])])
    for k2 in range(len(pairs)):
        row=[]
        for k3 in range(len(pairs)):
            row.append(((float(pairs[k2][0])-float(pairs[k3][0]))**2+(float(pairs[k2][1])-float(pairs[k3][1]))**2)**0.5)
        newmatrix.append(row)
    newmatrix = np.array(newmatrix)
    algorithm3 = SimulatedAnnealing(newmatrix.copy(), alpha,stopping_iter, listlambadesired2)
    print("START swaptour")
    swaptour = algorithm3.annealSwapLocations(quantities, maxquantity, betterdepottour.copy(),len(besttour))
    print("swaptour: ",swaptour)
    # with open(DATFILE, 'a') as f:
    #     f.write(str(algorithm3.weight(swaptour)) + "\n")
    #     f.write(str(swaptour)+ "\n")
    with open(DATFILE, 'w') as f:
         f.write(str(algorithm3.weight(swaptour)))
    print("besttour:", besttour)
    print("depottour:",depottour)
    print("betterdepottour:", betterdepottour)
    print("swaptour:", swaptour)
    print("length of besttour:", algorithm.weight(besttour))
    print("length of depottour:",algorithm.weight(depottour))
    print("length of betterdepottour:", algorithm.weight(betterdepottour))
    print("length of swaptour:",algorithm3.weight(swaptour))

    #algorithm.animateSolutions()
    # algorithm.showCoordinates(x_coords,y_coords,x_depot,y_depot,x_swap,y_swap)
    # algorithm.plotDepottour(besttour,pairs,x_coords,y_coords,x_depot,y_depot,x_swap,y_swap)
    # algorithm.plotDepottour(depottour, pairs,x_coords,y_coords,x_depot,y_depot,x_swap,y_swap)
    # algorithm.plotDepottour(betterdepottour, pairs,x_coords,y_coords,x_depot,y_depot,x_swap,y_swap)
    # algorithm.plotDepottour(swaptour, pairs, x_coords, y_coords, x_depot, y_depot, x_swap, y_swap)
    # algorithm.plotLearningWeightTime()
    # algorithm.plotLearningWeight()
    # algorithm.plotLearningTemp()
    # algorithm.plotLearningTime()
    #
    # algorithm2.plotLearningWeightTime()
    # algorithm2.plotLearningWeight()
    # algorithm2.plotLearningTemp()
    # algorithm2.plotLearningTime()
    return [algorithm3.weight(swaptour),swaptour]
# lengths = []
# tours= []
# for _ in range(1000):
#     outcome =test(0.95,50,"DATFILE",'locations.csv',16,4)
#     lengths.append(outcome[0])
#     tours.append(outcome[1])
# print(lengths)
# print(min(lengths))
# sorted = [x for _, x in sorted(zip(lengths, tours))]
# print(sorted[0])

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser("This program calculates minimal route for set of points")
    my_parser.add_argument('--alpha', type=float, required=True, help="Alpha value temperature")
    my_parser.add_argument('--stopping_iter', type=int , required=True, help="Max number of runs")
    my_parser.add_argument('--lambda1', type=int, required=True, help="Desired acceptance during run")
    my_parser.add_argument('--lambda2', type=int, required=True, help="Desired acceptance during run")
    my_parser.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
    my_parser.add_argument('--location', dest='location', type=str, required=True,
                           help='File containing the tour location description')
    my_parser.add_argument('--logfile', dest='logfile', type=str, required=True, help="Log file")


    class Unbuffered(object):
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()

        def writelines(self, datas):
            self.stream.writelines(datas)
            self.stream.flush()

        def __getattr__(self, attr):
            return getattr(self.stream, attr)


    import sys


    args = my_parser.parse_args()
    os.chdir("/home/robinv/Desktop/Capita/Capita")
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(args.logfile, 'w') as f:
        sys.stdout = Unbuffered(f)  # Change the standard output to the file we created.
         # Reset the standard output to its original value

        print("Start of test with alpha: ", args.alpha, "; Stopping_iter: ", args.stopping_iter, "; Locations: ", args.location)
        test(args.alpha, args.stopping_iter, args.datfile, args.location,args.lambda1,args.lambda2)
        sys.stdout = original_stdout

