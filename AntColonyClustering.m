clear;clc;
samples = csvread('iris.csv');
%samples = csvread('wine.csv');
%samples = csvread('thyroid.csv');
%samples = csvread('haberman.csv');

total_samples = size(samples, 1);
total_attributes = size(samples, 2);
total_clusters = 3;
total_agents = 50;

iteration = 1;
MAX_ITERATION = 2000;

RHO = 0.5;
p1s = 0.01;
q0 = 0.98;
tao = ones(total_samples, total_clusters) * 0.001;

local_search_candidate_ratio = 0.2;
total_top_solutions = round(total_agents * local_search_candidate_ratio);

top_global_solutions = [];
normalized_tao = [];
best_fitness = zeros(MAX_ITERATION, 1);
best_fitness_each_iter = zeros(MAX_ITERATION, 1);

while iteration <= MAX_ITERATION
    normalized_tao = tao./repmat(sum(tao, 2), 1, total_clusters);
    solutions = zeros(total_agents, total_samples+1); % solutions: total agents x total samples
    top_solutions = [];
    
    for i = 1:total_agents
        % GENERATE SOLUTIONS
        for N = 1:total_samples
            randomNum = rand();
            if randomNum > q0 % DO EXPLOITATION: using probability
                cummulative = 0;
                randomNum2 = rand();
                range = sort(normalized_tao(N, :));
                for x = 1:size(range, 2)
                   cummulative = cummulative + range(x);
                   if randomNum2 < cummulative
                       founded = find(normalized_tao(N, :) == range(x));
                       if size(founded, 2) == 1
                           solutions(i, N) = founded;
                       else
                           solutions(i, N) = founded(floor(rand()*size(founded, 2) + 1));
                       end
                       break
                   end
                end
            else % DO EXPLORATION: highest pheromone
                founded = find(tao(N, :) == max(tao(N, :)));
                if size(founded, 2) == 1
                    solutions(i, N) = founded;
                else
                    solutions(i, N) = founded(floor(rand()*size(founded, 2) + 1));
                end
            end
        end % END OF GENERATE SOLUTIONS
        
        % WEIGHT of i-th solution
        weight = zeros(total_samples, total_clusters);
        for N = 1:total_samples
            weight(N, solutions(i, N)) = 1;
        end
        
        % CENTROID of i-th solution
        centroid = (weight' * samples) ./ repmat(sum(weight, 1)', 1, total_attributes);
        % FITNESS of i-th solution
        for N = 1:total_samples
            solutions(i, end) = solutions(i, end) + sqrt(sum((samples(N, :) - centroid(solutions(i, N), :)).^2));
            %solutions(i, end) = solutions(i, end) + (sum((samples(N, :) - centroid(solutions(i, N), :)).^2));
        end
    end

    % LOCAL SEARCH
    top_solutions = sortrows(solutions, size(solutions, 2));
    top_solutions = top_solutions(1:total_top_solutions, :);
    temporary_top_solutions = top_solutions;

    for L = 1:total_top_solutions
        isUpdate = 0;
        for N = 1:total_samples
            if rand() <= p1s
                % random outside that cluster
                oldSolution = temporary_top_solutions(L, N);
                while temporary_top_solutions(L, N) == oldSolution
                    temporary_top_solutions(L, N) = floor(rand()*total_clusters)+1;
                end
                isUpdate = 1;
            end
        end

        if isUpdate == 1
            % WEIGHT of L-th top solutions
            weight1 = zeros(total_samples, total_clusters);
            for N = 1:total_samples
                weight1(N, temporary_top_solutions(L, N)) = 1;
            end
            % CENTROID of L-th top solutions
            temporary_top_solutions_centroid = (weight1' * samples)./ repmat(sum(weight1, 1)', 1, total_attributes);
            % FITNESS of L-th top solutions
            newFitness = 0;
            for N = 1:total_samples
                newFitness = newFitness + sqrt(sum((samples(N, :) - temporary_top_solutions_centroid(temporary_top_solutions(L, N), :)).^2));
                %newFitness = newFitness + (sum((samples(N, :) - temporary_top_solutions_centroid(temporary_top_solutions(L, N), :)).^2));
            end
            % UPDATE IF BETTER
            if newFitness < temporary_top_solutions(L, end)
                temporary_top_solutions(L, end) = newFitness;
                top_solutions(L, :) = temporary_top_solutions(L, :);
            end
        end

        % if it is better, update global top solution
        temporary_top_solutions = sortrows(temporary_top_solutions, size(temporary_top_solutions, 2));
        if iteration == 1
            top_global_solutions = temporary_top_solutions(1, :);
        end
        if temporary_top_solutions(1, end) < top_global_solutions(1, end)
            top_global_solutions = temporary_top_solutions(1, :);
        end
    end

    % UPDATE PHEROMONE -> tao = (1-RHO).*tao;
    for L = 1:total_top_solutions
        for N = 1:total_samples
            tao(N, top_solutions(L, N)) = (1-RHO).*tao(N, top_solutions(L, N)) + (1./top_solutions(L, end));
        end
    end
    
    fprintf('iteration: %d/%d | best fitness of all iteration: %d\n', iteration, MAX_ITERATION, top_global_solutions(1, end));
    best_fitness(iteration, 1) = top_global_solutions(1, end);
    best_fitness_each_iter(iteration, 1) = top_solutions(1, end);
    
    iteration = iteration + 1;
end

close
f = figure;
subplot(2, 2, 1);
plot([1:MAX_ITERATION]', best_fitness_each_iter(:, 1)');
title('Best Fitness Each Iteration');
xlabel('Iterations');
ylabel('Best Fitness Each Iteration');

subplot(2, 2, 2);
plot([1:MAX_ITERATION]', best_fitness(:, 1)');
title('Best Fitness All Iteration');
xlabel('Iterations');
ylabel('Best Fitness');

subplot(2, 2, 3);
histogram(top_global_solutions(1:end-1));
title('Cluster Result');
xlabel('Cluster');
ylabel('Total Samples');

subplot(2, 2, 4);
x = samples(:, 1);
y = samples(:, 2);
z = samples(:, 3);
scatter3(x, y, z, 20, top_global_solutions(1, 1:end-1));
title('Cluster Plot');
xlabel('Sepal Length');
ylabel('Sepal Width');
xlabel('Petal Length');

set(f, 'position', [200, 200, 700, 700]);
drawnow