
import matplotlib.pyplot as plt

def plot_stats_1(df, name='topic_overview_1.pdf'):

	Xax = df.loc[:,'average_n_cit_internal'].to_numpy()/df.loc[:, 'average_n_cit'].to_numpy()
	Yax = df.loc[:,'n_doc_over_expressed_topj'].to_numpy()/df.loc[:,'n_doc_over_expressed'].to_numpy()
	mean_Y = np.mean(Yax)
	mean_X = np.mean(Xax)

	plt.figure(figsize=(8,8))
	plt.grid(True)#,alpha=0.5)

	sizes = df.loc[:,'n_doc_over_expressed']*10

	plt.scatter(Xax,Yax, s = sizes)#, c = colors, cmap = 'viridis')
	plt.axvline(x=mean_X)
	plt.axhline(y=mean_Y)


	#for n,i,j in zip(df['Topic'],Xax,Yax):
	#    if n in [9]:
	#        plt.text(x=i,y=j,s=str(n))
	#    else:
	#        plt.text(x=i+0.0005,y=j+0.001,s=str(n))#df_2.iloc[n,0]))

	plt.xlabel('Ratio citations')
	plt.ylabel('Ratio top journals')
	plt.title('Topic Overview')

	plt.savefig(name,dpi=300)#,transparent = True)
	plt.show()

	return

def plot_stats_2(df, name='topic_overview_2.pdf'):

	Xax =df['average_n_cit_internal']
	Yax = df['average_n_cit']
	mean_Y = np.mean(Yax)
	mean_X = np.mean(Xax)

	plt.figure(figsize=(8,8))
	plt.grid(True)#,alpha=0.5)

	sizes = df.loc[:,'n_doc_over_expressed']*10

	plt.scatter(Xax, Yax, label='-', s = size, alpha=0.4)
	plt.axvline(x=mean_X)
	plt.axhline(y=mean_Y)


	#for n,i,j in zip(df['Topic'],Xax,Yax):
	#    if n in [9]:
	#        plt.text(x=i,y=j,s=str(n))
	#    else:
	#        plt.text(x=i+0.0005,y=j+0.001,s=str(n))#df_2.iloc[n,0]))

	plt.xlabel('Internal citations')
	plt.ylabel('Overall citations')
	plt.title('Topic Overview Citations')

	plt.savefig(name,dpi=300)#,transparent = True)
	plt.show()

	return



#plot_stats_1(df, nameplot1='topic_overview_1.pdf')
#plot_stats_2(df, nameplot2='topic_overview_2.pdf')




