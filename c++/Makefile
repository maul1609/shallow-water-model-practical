CXX = g++
RM = rm -rf

OBJECTS = main.o

shallow_water_model: $(OBJECTS)
	$(CXX) -o shallow_water_model $^

main.o: main.cpp
	$(CXX) -c $<

clean:
	$(RM) shallow_water_model $(OBJECTS)
