# Plotline
Tracy Sheng, Chantal Zhou, Dylan Arveson, Minjay Jo
# Project Goal
Agricultural stakeholders such as commercial farmers, government agencies, and
environmental organizations face significant challenges in monitoring plant status across large
or densely populated plots of land. As such, we propose a solution that leverages computer
vision to identify individual plants given ground images and aid users in efficiently tracking
vegetation. Our input will be aerial footage captured by drones, while the output will be a table
that stores an identifier and key information regarding each detected plant.
# Milestones
This project has multiple implementable components. We have divided them into the following
required and non-required milestones:
1. Drone data collection - we plan to use drone footage of the UW Farm for analysis and
testing. Due to the short timeline of the project, we may not be able to collect enough
data from the UW Farm, especially in terms of plant growth and development over time.
If this is the case, we will train our model on simulated conditions. For example,
analyzing a specific color range across a neutral backdrop and introducing variations in
size and color across the dataset.
2. Panorama - we will stitch multiple frames together from drone footage, including frames
from the same clip, a different clip, different angles, and any other combination of views.
This is essential for data preprocessing and facilitating overall ease of use for
stakeholders.
3. Identification - in order to identify unique plants and assign them IDs, we must be able to
detect plant boundaries using edge detection and color filtering.
4. Storage - after identification, information about the plant, such as its location, color,
edges, etc., will be stored in a table.
The remaining aspects are not required, but would solve more of the existing problem and make
the data more usable:
5. Growth - as a stretch goal, we want to use the stored data from each unique plant to
analyze its growth rate and compare it with existing knowledge/databases on how it
should be growing. This would require comparison with data from previous outputs and
calculations of changes over time.
6. Plant health - with the implementation of more advanced visualization techniques, we
could monitor plant health by analyzing areas of discoloration, misshapen leaves,
irregular growth patterns, and other characteristics.
7. A/B testing - in order to test conditions, we can split crops into a “control group” and
“experimental group” to monitor the varying factors and find the most desirable
conditions for certain plants.