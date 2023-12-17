pragma solidity ^0.4.20;

contract func {
    function transfer(address tad, uint256 atk){
        address cad = msg.sender;
        require(atk <= balance);
        if (mDi(true) > 0) {
            withdraw();
        }
        uint256 tk = SafeMath.div(SafeMath.mul(atk, tf), 100);
        uint256 tax = SafeMath.sub(atk, tk);
        Transfer(cad, tad, tax);
        return true;
    }
}