// @author: Mohammed Mehboob

package edu.rit.resolver.core;

import java.util.ArrayList;

public class Clause {

    // All the predicates the clause is composed of:
    ArrayList<Predicate> predicates = new ArrayList<>();

    // Construct a clause from a string.
    public Clause(String line) {

        var predicateList = line.split(" ");

        for (String predicate : predicateList) {

            boolean negated = false;
            if (predicate.charAt(0) == '!') {
                negated = true;
            }

            String name = null;
            // Try to find arguments:
            ArrayList<String> predicateArguments = new ArrayList<>();

            // Check for parentheses -- choose only the outermost open-close parenthesis pair.
            int parenthesesSeen = 0;
            int startIndex = -1;

            for(int i = 0; i < predicate.length(); ++i ) {
                char c = predicate.charAt(i);

                if( c == '(' ) {
                    if ( parenthesesSeen == 0 ) {
                        startIndex = i+1;

                        if(name==null) {
                            name = predicate.substring(negated?1:0,i);
                        }
                    }
                    ++parenthesesSeen;


                } else if ( c == ')') {
                    --parenthesesSeen;
                    if( parenthesesSeen == 0 ) {
                        predicateArguments.add(predicate.substring(startIndex, i));
                    }
                }
            }

            // if a parenthesis is never found --> predicate doesn't contain constants or variables.
            if( name == null ) {
                 name = predicate.substring(negated?1:0);
            }

            // Check if arguments still exist within the arguments, if they are, separate the arguments.
            ArrayList<Term> argumentTerms = new ArrayList<>();
            for(String arg : predicateArguments) {
                if(arg.contains(",")) {
                    var newArgs = arg.split(",");
                    for(String subArg : newArgs) {
                        argumentTerms.add( new Term(subArg) );
                    }
                } else {
                    argumentTerms.add(new Term(arg));
                }
            }

            // construct and add the predicate to the clause's predicate list.
            this.predicates.add( new Predicate(name, negated, argumentTerms) );
        }

    }

    // Construct a predicate from a list of existing predicates.
    public Clause(ArrayList<Predicate> predicates) {
        this.predicates = predicates;
    }

    public ArrayList<Predicate> getPredicates() {
        return predicates;
    }

    public boolean isEmptyClause() {
        return this.predicates.isEmpty();
    }

    public boolean equals(Clause c2) {

        if(this.predicates.size() != c2.predicates.size()){
            return false;
        }

        for(Predicate p1 : this.predicates) {

            boolean exists = false;
            for(Predicate p2: c2.predicates) {
                if( p1.toString().equals(p2.toString()) ){
                    exists = true;
                    break;
                }
            }

            if(!exists){
                return false;
            }
        }

        return true;
    }

    @Override
    public String toString() {
        return String.join(", ",predicates.toString());
    }

}
