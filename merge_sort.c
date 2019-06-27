#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define MASTER 0
#define TAG1 50
#define TAG2 60
#define TAG3 70
#define TAG4 80
#define TAG5 90
#define TAG6 100
#define TAG7 110
#define TAG8 120
#define TAG9 130

void sort(int *array, int start, int finish);
void merge(int *array,int start, int middle, int finish);
void init_array(int *array, int size, int lower, int upper);
void print_array(int *array, int size, char *str);
void allocationFailure(int *array);
int binary_search(int *array, int procRank, int start, int finish, int element, int *pos);

int main(int argc, char *argv[])
{
    int myrank, processors;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int *array, *part_array;
    int *lowB, *upperB, *tempArray, *helpArray;
    int size, core_count, tempSize, halfcores, helpSize, start, median, position, oddPosition, partial, split, source, target, overlap;
    double start_time, end_time;
    MPI_Status status;
    start = 0;
    
    if (myrank == MASTER) {
        puts("Give the size of the array: ");
        scanf("%d", &size);

        array = malloc(sizeof(int)*size);
        if (!array)
            allocationFailure(array);

        init_array(array, size, start, 99);
        start_time = MPI_Wtime();
        print_array(array,size, "-----------Initial array---------");
        core_count = processors;
        partial = size/core_count;
    }

    MPI_Bcast(&core_count, 1,MPI_INT, MASTER,MPI_COMM_WORLD);
    MPI_Bcast(&partial, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    MPI_Comm subComm;

    while (core_count > 1)
    {
        if (core_count == 1)
            break;

        int even_rank = myrank + 1;
        int odd_rank = myrank - 1;

        MPI_Comm_split(MPI_COMM_WORLD, core_count, myrank, &subComm);

        if (myrank < core_count)
        {
            part_array = malloc(sizeof(int)*partial);
            if(!part_array)
                allocationFailure(part_array);

            MPI_Scatter(array,partial, MPI_INT, part_array, partial, MPI_INT, MASTER, subComm);

            sort(part_array,start,partial-1);

            start=0;
            // Send the lower bounds to the MASTER
            if (myrank%2!=0)
            {
                MPI_Send(&part_array[start],1,MPI_INT,MASTER,TAG1,subComm);
            }
            // Send the upper bound
            else if (myrank%2==0)
            {
                MPI_Send(&part_array[partial-1],1,MPI_INT,MASTER,TAG2,subComm);
            }

            //Master receive the bounds and send them again to the proper destinations
            halfcores = core_count/2;
            if (myrank==MASTER)
            {
                lowB = malloc(sizeof(int)*halfcores);
                if(!lowB)
                    allocationFailure(lowB);
                upperB = malloc(sizeof(int)*halfcores);
                if(!upperB)
                    allocationFailure(upperB);

                for (source=1, start=0; source<core_count; source += 2, start++)
                    MPI_Recv(&lowB[start], halfcores, MPI_INT, source, TAG1, subComm, &status);   

                for (source=0, start=0; source<core_count; source+=2, start++)
                    MPI_Recv(&upperB[start], halfcores, MPI_INT, source, TAG2, subComm, &status);

                for (target=1, start=0; target<core_count; target+=2, start++)
                    MPI_Send(&upperB[start],1,MPI_INT,target,TAG3,subComm);

                for (target=0, start=0; target<core_count; target+=2, start++)
                    MPI_Send(&lowB[start],1,MPI_INT,target,TAG3,subComm);
            }

            MPI_Recv(&overlap, 1, MPI_INT, MASTER, TAG3, subComm, &status);
            printf("I am %d and my overlap is %d\n",myrank,overlap);
            
            start=0;
            split = binary_search(part_array, myrank, start, partial, overlap, &position);
            printf("I am processor %d and the split is: %d in position: %d\n",myrank, split, position);

            if (myrank%2!=0)
            {
                MPI_Send(&position,1,MPI_INT,odd_rank,TAG4,subComm);
                MPI_Send(&part_array[0],position,MPI_INT,odd_rank,TAG5,subComm);
            }
            else
            {
                MPI_Recv(&oddPosition,1,MPI_INT,even_rank,TAG4,subComm,&status);

                helpSize=(partial-position)+oddPosition;
                helpArray = malloc(sizeof(int)*helpSize);
                if(!tempArray)
                    allocationFailure(helpArray);

                for (int i=0, start=position; start<=partial-position; i++, start++)
                    helpArray[i] = part_array[start];

                MPI_Recv(&helpArray[helpSize-oddPosition],oddPosition,MPI_INT,even_rank,TAG5,subComm,&status);
                
                start=0;
                sort(helpArray, start, helpSize-1);
                
                median = (int) helpSize/2;
            
                for (int i=0, j=partial-median; i<median; i++, j++)
                    part_array[j] = helpArray[i];

                MPI_Send(&helpArray[median],oddPosition, MPI_INT, even_rank,TAG6,subComm);

            }

            if (myrank%2!=0)
            {
                MPI_Recv(&part_array[0],position,MPI_INT,odd_rank,TAG6,subComm,&status);
                MPI_Send(&part_array[0],partial,MPI_INT,odd_rank,TAG7,subComm);
            } 
            else 
            {

                tempSize = partial*2;
                tempArray = malloc(sizeof(int)*tempSize);
                if(!tempArray)
                    allocationFailure(tempArray);

                for (int i=0; i<partial; i++)
                    tempArray[i] = part_array[i];

                MPI_Recv(&tempArray[partial],partial,MPI_INT,even_rank,TAG7,subComm,&status);
                MPI_Send(&tempArray[0],tempSize,MPI_INT, MASTER, TAG8,subComm);

            }

            //MPI_Gather(part_array, partial, MPI_INT, array, size, MPI_INT, MASTER, subComm);

            if(myrank==MASTER)
            {
                start=0;
                for (source=0; source<core_count; source+=2)
                {

                    MPI_Recv(&array[start],tempSize, MPI_INT,source, TAG8, subComm,&status);
                    start += tempSize;
                }

                free(lowB);
                free(upperB);
            }

            if (myrank%2==0)
            {
                free(helpArray);
                free(tempArray);
            }

            core_count/=2;
            partial*=2;
            free(part_array);
        }
        MPI_Comm_free(&subComm);
    }

    if (myrank==MASTER)
    {
        end_time = MPI_Wtime();
        printf("The time elapsed is: %.16f\n\n", end_time - start_time);
        print_array(array, size, "---------Sorted Array------------");
    }

    free(array);
    MPI_Finalize();
    return (0);
}

void init_array(int *array, int size, int lower, int upper) {
    int i;
    //srand(time(NULL));

    for (i=0; i<size; i++)
        array[i] = lower+rand()%(upper-lower+1);
}


void print_array(int *array, int size, char *str) {
    int i;

    printf("%s\n",str);
    printf("[");
    for (i=0; i<size-1; i++)
        printf("%d, ", array[i]);
    printf("%d]\n", array[size-1]);
}

void sort(int *array,int start, int finish) {
    if (start < finish) {
        int middle = start + (finish-start)/2;
        sort(array,start, middle);
        sort(array,middle+1, finish);
        merge(array,start, middle, finish);
    }
}

void merge(int *array,int start, int middle, int finish) {
    int left, m, right, i, j;
    int tempArray[finish+1];
    m = middle;
    left = start;
    right = m+1;
    i=0;

    while (left <= m && right <= finish)
    {
        if (array[left] <= array[right]) {
            tempArray[i] = array[left];
            left++;
            i++;
        } else {
            tempArray[i] = array[right];
            right++;
            i++;
        }
    }

    if (left == m+1) {
        while (right <= finish) {
            tempArray[i] = array[right];
            right++;
            i++;
        }
    } else {
        while (left <= middle) {
            tempArray[i] = array[left];
            left++;
            i++;
        }
    }

    i = 0;
    j = start;

    while (j <= finish) {
        array[j] = tempArray[i];
        i++;
        j++;
    }
}

void allocationFailure(int *array) {
    puts("Failed to allocate memory!");
    exit(0);
}

int binary_search(int *array, int procRank, int start, int finish, int element, int *pos) {
    int middle;

    if (start>finish)
        return 0;
    else {
        middle=(1+start+finish)/2;
        if (element > array[middle] && element < array[middle+1])
        {
            *pos = middle+1;
            return array[middle];
        }
        else if (element < array[middle])
        {
            if (element <= array[start]) {
                *pos = start;
                return array[start];
            } else
                return binary_search(array, procRank, start, middle-1, element, pos);
        }
        else if (element > array[middle])
        {
            if (element >= array[finish]) {
                *pos = finish;
                return array[finish-1];
            }
            return binary_search(array, procRank, middle+1, finish, element, pos);
        }
        else  // situation where element == array[middle]
        {
            if (procRank%2!=0) {
                int i = middle-1;
                if (element > array[i]) {
                    *pos = middle+1;
                    return array[middle];
                }
                else {
                    while (i > start) {
                        if (element == array[i])
                            i--;
                        else {
                            *pos = i+1;
                            return array[i];
                        }
                    }
                }
                *pos = start;
                return array[start];
            }
            else {
                int i = middle+1;
                if (element < array[i]) {
                    *pos = middle+1;
                    return array[middle];
                }
                else {
                    while (i < finish) {
                        if (element == array[i])
                            i--;
                        else {
                            *pos = i+1;
                            return array[i];
                        }
                    }
                }
                *pos = finish;
                return array[finish-1];
            }
        }
    }
}

