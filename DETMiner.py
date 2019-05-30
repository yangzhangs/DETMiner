#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'yang'

import sys
import os
import MySQLdb
import numpy as np
from math import factorial
from numpy import vstack,array
import scipy.fftpack
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def drawPlot(x,y,x1,y1):
	plt.plot(x, y, 'go-')
	plt.plot(x1, y1, 'ro-')
	#plt.ylim(y.min()-1,y.max()+1)
	#plt.xlim(x.min()-1,x.max()+1)
	plt.xlabel('Periods')
	plt.ylabel('Dockerfile Size')
	plt.grid(True)
	#plt.legend(['data', 'linear', 'cubic'], loc='best')
	plt.show()


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
	
	
def dataResampling(y):
	#print len(y)
	f = np.interp(np.arange(0,len(y),float(len(y))/20),np.arange(0,len(y)),y)
	#plt.plot(x,f)
	#plt.show()
	#print len(f)
	return f

def dataSmoothing0(changes):
	length = len(changes)
	
	x = np.linspace(1, length, num=length, endpoint=True)
	y = np.array(changes)
	#print y
	#f = interp1d(x, y)
	f2 = interp1d(x, y, kind='cubic')
	xnew = np.linspace(1, length, num=2*length, endpoint=True)
	drawPlot(x,y)
	drawPlot(xnew,f2(xnew))
	y1 = f2(xnew)
	ynew = dataResampling(y1)
	return ynew

def dataSmoothing1(changes, box_pts):
	length = len(changes)
	x = np.linspace(1, length, num=length, endpoint=True)
	y = np.array(changes)
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	xnew = np.linspace(1, length, num=len(y_smooth), endpoint=True)
	drawPlot(x,y,xnew,y_smooth)
	
def dataSmoothing2(changes):
	length = len(changes)
	x = np.linspace(1, length, num=length, endpoint=True)
	y = np.array(changes)
	w = scipy.fftpack.rfft(y)
	f = scipy.fftpack.rfftfreq(length, x[1]-x[0])
	spectrum = w**2
	cutoff_idx = spectrum < (spectrum.max()/5)
	w2 = w.copy()
	w2[cutoff_idx] = 0
	y2 = scipy.fftpack.irfft(w2)
	xnew = np.linspace(1, length, num=len(y2), endpoint=True)
	drawPlot(x,y,xnew,y2)
	
def dataSmoothing(changes):
	length = len(changes)
	
	x = np.linspace(1, length, num=length, endpoint=True)
	y = np.array(changes)
	#print y
	#f = interp1d(x, y)
	f2 = savitzky_golay(y, 51, 3)
	xnew = np.linspace(1, length, num=len(f2), endpoint=True)
	drawPlot(x,y,xnew,f2)
	
def dataSmoothing3(changes):
	length = len(changes)
	x = np.linspace(1, length, num=length, endpoint=True)
	y = np.array(changes)
	kr = KernelReg(y,x,'c')
	r_fit =  KernelReg.r_squared(kr)
	#plt.figure(1)
	#plt.subplot(131)
	#plt.plot(x, y, 'go-')
	#plt.title("Original",fontsize=20)
	#plt.xlabel('Periods',fontsize=20)
	#plt.ylabel('Dockerfile Size',fontsize=20)
	#plt.grid(True)
	if length < 20:
		x1 = np.linspace(1, length, num=3*length, endpoint=True)
	else:
		x1 = x
	y_pred, y_std = kr.fit(x1)
	#plt.subplot(132)
	#plt.plot(x1, y_pred,'bo-')
	#plt.title("Smoothing",fontsize=20)
	#plt.xlabel('Periods',fontsize=20)
	#plt.ylabel('Dockerfile Size',fontsize=20)
	#plt.grid(True)
	#plt.show()
	ynew = dataResampling(y_pred)
	xnew = np.linspace(1, 20, 20, endpoint=False)
	#plt.subplot(133)
	#plt.plot(xnew, ynew,'ro-')
	#plt.title("Resampling",fontsize=20)
	#plt.xlabel('Periods',fontsize=20)
	#plt.ylabel('Dockerfile Size',fontsize=20)
	#plt.grid(True)
	#plt.show()
	return ynew,r_fit
	

def getProjectChanges(repo,cur):
	changes = []
	try:
		query = "select size from dockerfile_changes_100 where repo=\"%s\" and total>0 order by committer_date"%repo
		cur.execute(query)
		data = cur.fetchone()
		while data != None:
			changes.append(data[0])
			data = cur.fetchone()
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
	return dataSmoothing3(changes)


def getProjectInfo():
	conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur = conn.cursor()
	conn_1 = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur_1 = conn_1.cursor()
	conn_1.set_character_set('utf8')
	conn_2 = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur_2 = conn_2.cursor()
	try:
		query = "select repo,id from select_repos_100 where repo not in (select repo from change_points_4)"
		cur.execute(query)
		data = cur.fetchone()
		while data != None:
			print data[1],data[0]
			repo = data[0]
			points = ""
			sample_points,r_fit = getProjectChanges(repo,cur_1)
			min = sample_points.min()
			max = sample_points.max()
			if len(sample_points)==21:
				for i in range(0,len(sample_points)-1):
					x=(sample_points[i]-min)/(max-min)
					points = points+str(x)+","
			elif len(sample_points)==20:
				for i in range(0,len(sample_points)):
					x=(sample_points[i]-min)/(max-min)
					points = points+str(x)+","
			points = points+str(1)
			print points
			try:
   				query_2 = "insert into change_points_4(repo,period_1,period_2,period_3,period_4,period_5,"\
   	     		          "period_6,period_7,period_8,period_9,period_10,"\
   				          "period_11,period_12,period_13,period_14,period_15,"\
   				          "period_16,period_17,period_18,period_19,period_20,r_fit) values(\"%s\",%s)" %(repo,points)
   				cur_2.execute(query_2)
   		 	except MySQLdb.Error, e:
   				print "Mysql Error!", e;
			data = cur.fetchone()
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
	cur.close()
	conn.close()
	cur_1.close()
	conn_1.close()
	cur_2.close()
	conn_2.close()
	
def k_means(X):
	kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
	groups = kmeans.labels_
	return groups

def getPoints():
	conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur = conn.cursor()
	conn_1 = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur_1 = conn_1.cursor()
	conn_1.set_character_set('utf8')
	sets = []
	ids = []
	try:
		query = "select * from change_points_4 where repo in (select repo from final_projects)"
		cur.execute(query)
		data = cur.fetchone()
		while data != None:
			print data[0]
			repo = data[1]
			points = []
			for i in range(2,22):
				points.append(data[i])
			sets.append(points)
			ids.append(data[0])
			data = cur.fetchone()
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
	groups = k_means(sets)
	for j in range(0,len(ids)):
		try:
   			query_1 = "update change_points_4 set k_group_1=\"%s\" where id=%s" %("Cluster-"+str(groups[j]+1),ids[j])
   			#print query_1
   			cur_1.execute(query_1)
   		except MySQLdb.Error, e:
   			print "Mysql Error!", e;
	cur.close()
	conn.close()
	cur_1.close()
	conn_1.close()

def caseStudy():
	conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur = conn.cursor()
	sets = []
	try:
		query = "select * from change_points_4 where k_group=6 order by rand() limit 5"
		cur.execute(query)
		data = cur.fetchone()
		while data != None:
			#print data[0]
			repo = data[1]
			points = []
			for i in range(2,22):
				points.append(data[i])
			sets.append(points)
			data = cur.fetchone()
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
	plt.figure(4)
	x = np.linspace(1, 20, 20, endpoint=False)
	mins = []
	maxs = []
	sets = np.array(sets)
	for set in sets:
		plt.plot(x, set, 'bo-')
		mins.append(set.min())
		maxs.append(set.max())
	mins = np.array(mins)
	maxs = np.array(maxs)
	plt.ylim(mins.min()-0.2,maxs.max()+0.2)
	plt.xlim(x.min()-1,x.max()+1)
	#plt.title("Group-1",fontsize=20)
	plt.xlabel('Periods',fontsize=20)
	plt.ylabel('Dockerfile Size',fontsize=20)
	plt.grid(True)
	plt.show()
	cur.close()
	conn.close()

def releaseFreq(repo,cur):
	try:
		query = "select 3600*24*30*count(*)/timestampdiff(second,min(created_date),max(created_date)) from all_dockerhub_builds where repo=\"%s\" and status>=0"%repo
		cur.execute(query)
		data = cur.fetchone()
		if data[0] != None:
			return data[0]
		else:
			return -1
	except MySQLdb.Error, e:
		print "Mysql Error!", e;

def totalBuilds(repo,cur):
	try:
		query = "select count(*) from all_dockerhub_builds where repo=\"%s\""%repo
		cur.execute(query)
		data = cur.fetchone()
		if data[0] != None:
			return data[0]
		else:
			return -1
	except MySQLdb.Error, e:
		print "Mysql Error!", e;

def errorBuilds(repo,cur):
	total = totalBuilds(repo,cur)
	try:
		query = "select count(*) from all_dockerhub_builds where repo=\"%s\" and status>=0"%repo
		cur.execute(query)
		data = cur.fetchone()
		if data[0] != None:
			return float(total-data[0])/total
		else:
			return -1
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
		
def buildLatency(repo,cur):
	try:
		query = "select avg(timestampdiff(second,created_date,last_updated)) from all_dockerhub_builds where repo=\"%s\" and status>=0"%repo
		cur.execute(query)
		data = cur.fetchone()
		if data[0] != None:
			return data[0]
		else:
			return -1
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
		
def dockerfileChanges(repo,cur):
	try:
		query = "select count(*) from (select distinct(committer_date) from dockerfile_changes_100 where repo=\"%s\")as t"%repo
		cur.execute(query)
		data = cur.fetchone()
		if data[0] != None:
			return data[0]
		else:
			return -1
	except MySQLdb.Error, e:
		print "Mysql Error!", e;

def updateInfo():
	conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur = conn.cursor()
	conn_1 = MySQLdb.connect(host='localhost',user='root',passwd='',db='cd')
	cur_1 = conn_1.cursor()
	conn_2 = MySQLdb.connect(host='localhost',user='root',passwd='',db='dockerfile')
	cur_2 = conn_2.cursor()
	conn_2.set_character_set('utf8')
	sets = []
	ids = []
	try:
		query = "select r2.dockerhub_repo,r1.id,r1.repo from change_points_1 r1,selected_repos r2 where r1.repo=r2.github_repo group by r1.repo"
		cur.execute(query)
		data = cur.fetchone()
		while data != None:
			print data[1]
			repo = data[0]
			release_freq = releaseFreq(repo,cur_1)
			error_builds = errorBuilds(repo,cur_1)
			build_latency = buildLatency(repo,cur_1)
			changes = dockerfileChanges(data[2],cur_2)
			total_builds = totalBuilds(repo,cur_1)
			try:
   				query_2 = "update change_points_1 set total_builds=%s,avg_build_latency=%s,error_builds=%s,release_freq=%s,changes=%s where id=%s" \
   							%(total_builds,build_latency,error_builds,release_freq,changes,data[1])
   				cur_2.execute(query_2)
   		  	except MySQLdb.Error, e:
   			   	print "Mysql Error!", e;
			data = cur.fetchone()
	except MySQLdb.Error, e:
		print "Mysql Error!", e;
	cur.close()
	conn.close()
	cur_1.close()
	conn_1.close()
	cur_2.close()
	conn_2.close()
	
def justPlot():
	x = np.linspace(1, 10, num=10, endpoint=True)
	#print x
	y = np.array([0,0.5856648,0.4606585,0.4454593,0.372822,0.3383765,0.3238148,0.2961258,0.289679,0.3132629])
	plt.plot(x,y,"ko-")
	plt.xlabel('K',fontsize=20)
	plt.ylabel('Mean Silhouette Score',fontsize=20)
	plt.ylim(0,0.8)
	plt.xlim(0.5,10.5)
	plt.annotate('0.59', xy=(2, 0.6), xytext=(3, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.03),
            )
	plt.annotate('0.45', xy=(4, 0.45), xytext=(5, 0.6),
            arrowprops=dict(facecolor='black', shrink=0.03),
            )
	plt.grid(True)
	plt.show()
	
if __name__ == '__main__':
	#getProjectInfo()
	getPoints()
	#caseStudy()
	#updateInfo()
	#justPlot()
