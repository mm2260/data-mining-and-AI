// @author: Mohammed Mehboob

package edu.rit.resolver;

import edu.rit.resolver.core.Clause;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Runner {

    private static void run(String filename) {

        ArrayList<Clause> clauses = new ArrayList<>();

        try {
            Scanner fileScanner = new Scanner(new File(filename));   // args[0] --> filename (.cnf)

            String[] predicatesInput = fileScanner.nextLine().split(" ");
            Resolver.predicates.addAll( Arrays.asList(predicatesInput).subList(1, predicatesInput.length) );

            String[] variablesInput = fileScanner.nextLine().split(" ");
            Resolver.variables.addAll( Arrays.asList(variablesInput).subList(1, variablesInput.length) );

            String[] constantsInput = fileScanner.nextLine().split(" ");
            Resolver.constants.addAll( Arrays.asList(constantsInput).subList(1, constantsInput.length) );

            String[] functionsInput = fileScanner.nextLine().split(" ");
            Resolver.functions.addAll( Arrays.asList(functionsInput).subList(1, functionsInput.length) );

            fileScanner.nextLine();
            while(fileScanner.hasNext()) {
                clauses.add( new Clause(fileScanner.nextLine()) );
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println( Resolver.resolution(clauses)?"yes":"no" );
    }

    public static void main(String[] args) {

        String filename = args[0];
        run(filename);
    }

}
