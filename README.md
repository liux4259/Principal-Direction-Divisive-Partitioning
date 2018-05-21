COPYRIGHT -- See end of this file.

This repo implements Principal Direction Divisive Partitioning (PDDP) in Python as described in 
https://link.springer.com/article/10.1023/A:1009740529316

LIST OF FILES
--------------------
PDDP.py: contains the algorithm to perform PDDP with different splitting criteria. The cluster selection criteria is fixed:
         it chooses the cluster with largest scatter value (Frobenius norm) to split next 
PDDPNode.py: contains the definition of PDDP node, which is used to form a PDDP tree
selection_criteria.py: contains methods for variants on how to select next cluster to split
spliting_criteria.py: contains methods and their helper methods for variants on how to split a selected cluster
helpers.py: contains helper methods used throughout the repo 

HOW TO USE
--------------------
You can either import pddp from PDDP.py to use it directly in your application. The PDDP.py also includes a main method, which 
can be used by command line. The command arguments are as follows:

python PDDP.py input_file output_file spliting_criteria document_representation num_clusters

An example command line to run the algorithm is as follows:

python PDDP.py matrix.in matrix.out origin tfidf 20

The available arguments for spliting_criteria include: origin, gap, kmeans, scatter, scatter_project, even.
The available arguments for representations cindlue: tfidf, norm_freq


% COPYRIGHT DANIEL BOLEY 1999,2000,2001,2003 ALL RIGHTS RESERVED.
% LICENSE TO USE THIS SOFTWARE IS SUBJECT TO TERMS IN THIS FILE.
%
% This software may be used only for non-commercial research purposes.
% It may not be redistributed for any fee, or included in any package which
% is not distributed for free.  It may not be included or used as part of
% any commercial web site.  Any non-commercial use or copy of this software
% must include these terms intact and an acknowledgement that such use or
% copy is by the permission and cooperation of Daniel Boley.  This software
% is provided as is.  No warranty is made that this software will work or
% meet any particular requirements, and no customer service or software
% maintenance will be provided.  By using or copying this software, you agree
% to these conditions.  Any exceptions to these conditions must be agreed to
% by all partied in writing % with a real physical signature (not faxed, not
% photocopied, not electronic).
