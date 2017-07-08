/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Set the number of particles
    num_particles = 256;
    weights.resize(num_particles);
    default_random_engine gen;
    
    // Gaussian noise
    normal_distribution<double> noise_x_init(x, std[0]);
    normal_distribution<double> noise_y_init(y, std[1]);
    normal_distribution<double> noise_theta_init(theta, std[2]);
    
    // Initialize all particles to first position
    for (int i = 0; i < num_particles; i++) {
        
        Particle new_particle;
        new_particle.id = i;
        
        // Add noise
        new_particle.x = noise_x_init(gen);
        new_particle.y = noise_y_init(gen);
        new_particle.theta = noise_theta_init(gen);
        
        // Initialize all weights to 1
        new_particle.weight = 1.0;
        
        particles.push_back(new_particle);
    }
    
    cout << "Number of Weights: " << weights.size() << endl;
    
    is_initialized = true;
    
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    
    // Gaussian noise
    normal_distribution<double> noise_x(0.0, std_pos[0]);
    normal_distribution<double> noise_y(0.0, std_pos[1]);
    normal_distribution<double> noise_theta(0.0, std_pos[2]);
    
    // Safety net
    if (fabs(yaw_rate) < 0.0001) {
        yaw_rate = 0.0001;
    }
    
    // Calculate new state + noise
    for (auto&& particle : particles){
        
        particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) + noise_x(gen);
        particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) + noise_y(gen);
        particle.theta += yaw_rate * delta_t + noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    vector<LandmarkObs> associations;
    
    for (int i=0; i < observations.size(); i++) {

        // Current observation
        LandmarkObs obs = observations[i];
        LandmarkObs min = predicted[0];
        
        double min_distance_squared = pow(min.x - obs.x, 2) + pow(min.y - obs.y, 2);
        
        for (int j=0; j < predicted.size(); j++) {
            double distance_squared = pow(predicted[j].x - obs.x, 2) + pow(predicted[j].y - obs.y, 2);
            if (distance_squared < min_distance_squared) {
                min = predicted[j];
            }
        }
        associations.push_back(min);
    }
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double var_x = pow(std_landmark[0], 2);
    double var_y = pow(std_landmark[1], 2);
    double covar_xy = std_landmark[0] * std_landmark[1];
    double weights_sum = 0;
    
    for (int i=0; i < num_particles; i++) {
        Particle& particle = particles[i];
        
        long double weight = 1;
        
        for (int j=0; j < observations.size(); j++) {
            LandmarkObs obs = observations[j];
            
            double predicted_x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            double predicted_y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            
            Map::single_landmark_s nearest_landmark;
            double min_distance = sensor_range;
            double distance = 0;
            
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                
                Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
                
                distance = fabs(predicted_x - landmark.x_f) + fabs(predicted_y - landmark.y_f);
                
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_landmark = landmark;
                }
                
                
            }
            
            double x_diff = predicted_x - nearest_landmark.x_f;
            double y_diff = predicted_y - nearest_landmark.y_f;
            double num = exp(-0.5*((x_diff * x_diff)/var_x + (y_diff * y_diff)/var_y));
            double denom = 2*M_PI*covar_xy;

            weight *= num/denom;
            
        }
        
        cout << "weight: " << weight << endl;
        
        particle.weight = weight;
        weights[i] = weight;
        weights_sum += weight;
        
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    
    discrete_distribution<> weights_pmf(weights.begin(), weights.end());

    vector<Particle> newParticles;

    for (int i = 0; i < num_particles; ++i)
        newParticles.push_back(particles[weights_pmf(gen)]);
    
    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
