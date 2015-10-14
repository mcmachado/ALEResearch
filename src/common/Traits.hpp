/**
 * @file   Traits.hpp
 * @date   Mon Oct 13 11:31:38 2014
 * @author Nicolas Carion
 * @brief  This file contains useful traits to check some properties of class 
 * (namely the implementation of static members and attributes). Some macros
 * to help the user do that are defined here.
 *
 *
 */

#ifndef TRAITS_HPP_INCLUDED
#define TRAITS_HPP_INCLUDED

#include <type_traits>
//#include <typeinfo>
#include <tuple>

namespace function_traits{

    namespace _implem{

        /** The following template provide access to the signature of a method of a class 
         * 
         */
        template<class F>
        struct function_traits;

        // function pointer
        template<class R, class... Args>
        struct function_traits<R(*)(Args...)> : public function_traits<R(Args...)>
        {};

        //we allow also const methods
        template<class T,class R, class... Args>
        struct function_traits<R (T::*)(Args...) const> : public function_traits<R (T::*)(Args...)>
        {};

        
        template<class T,class R, class... Args>
        struct function_traits<R (T::*)(Args...)>
        {
            using return_type = R;

            static constexpr std::size_t arity = sizeof...(Args);

            template <std::size_t N>
            struct get_arg
            {
                static_assert(N < arity, "error: invalid parameter index.");
                using type = typename std::tuple_element<N,std::tuple<Args...>>::type;
            };
        };



        /** This class checks that the signature of a given function  
         * matches a given pattern (we only check arguments)
         * 
         * @param pos Number of the parameter we are checking (initially 0)
         * @param T The type of the function to check
         * @param Args The list of the types of the arguments of the function that we want to ensure
         */
        template<int pos,typename T, typename... Args>
        struct check_signature;


        template<int pos,typename T, typename Arg>
        struct check_signature<pos,T,Arg>
        {
            using Traits = typename function_traits<T>::template get_arg<pos>;
            static constexpr bool value = std::is_same<Arg,typename Traits::type >::value;
        };

        template<int pos,typename T, typename Head, typename... Tail>
        struct check_signature<pos,T,Head,Tail...>
        {
            using Traits = typename function_traits<T>::template get_arg<pos>;
            static constexpr bool value = std::is_same<Head,typename Traits::type >::value && check_signature<pos+1,T,Tail...>::value;
        };


        /**This class is a lazy logical AND between its arguments
         * 
         * 
         */

        template<bool b1,bool b2>
        struct static_and
        {
            static constexpr bool value = false;
        };
        template<>
        struct static_and<true,true>
        {
            static constexpr bool value = true;
        };

    } //namespace _implem

}

/** This macro creates a template that allows to check whether the public method X is implemented with a given signature. 
 *
 * For example, if we want to insure that a given class C has a public method foo with signature "bool foo(float,int)", we have to first call the macro "CREATE_MEMBER_DETECTOR(foo)" and then we can use the class in a static assert :
 * "static_assert(has_foo<foo,bool(float,int)>::value,"error")
 */
#define CREATE_MEMBER_DETECTOR(X)                                       \
   namespace function_traits{template<typename C,typename F>            \
    class has_##X;                                                      \
                                                                        \
    template<typename C, typename Ret, typename... Args>                \
    class has_##X<C,Ret(Args...)> {                                     \
    private:                                                            \
    template<class U, class P = typename std::enable_if<function_traits::_implem::static_and< \
                                                                                              std::is_same< decltype( std::declval<U>().X( std::declval<Args>()... ) ), Ret >::value, \
                                                                                              function_traits::_implem::check_signature<0,decltype(&U::X),Args...>::value \
                                                                                              >::value>::type> \
    static std::true_type check(int);                                   \
    template <class,class P = void>                                     \
    static std::false_type check(...);                                  \
    public:                                                             \
    static  constexpr bool value = decltype(check<C>(0))::value;        \
    };}

/** This macro creates a template that allows to check whether the class has a static member called X. 

 */
#define CREATE_STATIC_MEMBER_DETECTOR(X)                                \
    namespace function_traits{template <class T>                        \
        class has_##X                                                   \
    {                                                                   \
        template<class U, class = typename std::enable_if<!std::is_member_pointer<decltype(&U::X)>::value>::type> \
                 static std::true_type check(int);                      \
        template <class>                                                \
                 static std::false_type check(...);                     \
    public:                                                             \
        static constexpr bool value = decltype(check<T>(0))::value;     \
    };}


#endif
